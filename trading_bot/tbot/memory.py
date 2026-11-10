"""Память и самообучение бота.

Три механизма:
1. Точность агентов. Каждый сигнал через N циклов сверяется с тем, куда реально пошла цена.
   Агенты, которые чаще угадывают, получают больший вес в голосовании.
2. Память похожих ситуаций. Каждая ситуация на рынке (RSI, импульс, объём, положение в канале,
   давление в стакане...) сохраняется вместе с тем, что случилось с ценой потом. В новой ситуации бот
   находит 20 самых похожих из прошлого и смотрит: как часто цена росла и насколько.
   У TradingAgents память - это просто последние решения; здесь - поиск по похожести и цифры.
3. Уроки сделок. Когда позиция закрыта, бот записывает вывод (через Claude или по шаблону) вместе с
   «отпечатком» ситуации входа. Трейдер видит уроки из самых похожих прошлых ситуаций.
"""
from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from .llm import LLM

ANALYSTS = ["technical", "news", "sentiment", "memory", "debate"]

# признаки ситуации и их масштаб (чтобы все были примерно одного порядка)
FEATURES = {
    "rsi": 50.0, "ret_5": 0.01, "ret_20": 0.02, "macd": 0.002, "volume_z": 2.0,
    "bb_pos": 0.5, "up_volume": 0.25, "book": 0.5, "trend": 0.01,
}


class Lesson(BaseModel):
    lesson: str = Field(description="Один практический вывод на будущее, 1-2 предложения, по-русски")


def situation(indicators: dict, behavior: dict, book: dict | None = None) -> dict:
    """«Отпечаток» рыночной ситуации - числа, по которым ищем похожие случаи."""
    i, b = indicators, behavior
    close = i.get("close") or 1.0
    bb_low, bb_high = i.get("bb_low"), i.get("bb_high")
    bb_pos = (close - bb_low) / (bb_high - bb_low) if bb_low and bb_high and bb_high > bb_low else 0.5
    trend = (i["ema20"] / i["ema50"] - 1) if i.get("ema20") and i.get("ema50") else 0.0

    def num(x, default=0.0):
        return default if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)

    return {
        "rsi": num(i.get("rsi14"), 50) - 50,
        "ret_5": num(i.get("ret_5")),
        "ret_20": num(i.get("ret_20")),
        "macd": num(i.get("macd_hist")) / close,
        "volume_z": num(i.get("volume_z")),
        "bb_pos": bb_pos - 0.5,
        "up_volume": num(b.get("up_volume_share"), 0.5) - 0.5,
        "book": num((book or {}).get("imbalance")) if (book or {}).get("available") else 0.0,
        "trend": trend,
    }


def _vec(f: dict) -> np.ndarray:
    return np.array([np.clip(f.get(k, 0.0) / s, -5, 5) for k, s in FEATURES.items()])


class Memory:
    def __init__(self, path: str | Path | None = None, eval_after_cycles: int = 5, min_samples: int = 10,
                 max_situations: int = 20_000):
        self.path = Path(path) if path else None
        self.eval_after = eval_after_cycles
        self.min_samples = min_samples
        self.max_situations = max_situations
        self.pending: list[dict] = []           # сигналы, ждущие проверки
        self.stats: dict[str, dict] = {a: {"hits": 0, "total": 0, "acc": 0.5} for a in ANALYSTS}
        self.situations: list[dict] = []        # {"f": признаки, "ret": что было потом, "interval", "ticker"}
        self.lessons: list[dict] = []           # {"text", "f", "ticker", "pnl_pct"}
        self.decisions: dict[str, dict] = {}    # decision_id -> краткая запись решения
        self.processed: list[str] = []         # ключи уже разобранных сделок
        # быстрый индекс для поиска похожих: буфер с удвоением ёмкости, без пересборки каждый цикл
        self._mat = np.zeros((256, len(FEATURES)))
        self._rets = np.zeros(256)
        self._ints: list = []
        if self.path and self.path.exists():
            self._load()
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        n = len(self.situations)
        cap = max(256, 2 * n)
        self._mat = np.zeros((cap, len(FEATURES)))
        self._rets = np.zeros(cap)
        for j, sit in enumerate(self.situations):
            self._mat[j] = _vec(sit["f"])
            self._rets[j] = sit["ret"]
        self._ints = [sit.get("interval") for sit in self.situations]

    def _add_situation(self, rec: dict) -> None:
        n = len(self.situations)
        if n >= len(self._rets):
            self._mat = np.vstack([self._mat, np.zeros_like(self._mat)])
            self._rets = np.concatenate([self._rets, np.zeros_like(self._rets)])
        self._mat[n] = _vec(rec["f"])
        self._rets[n] = rec["ret"]
        self._ints.append(rec.get("interval"))
        self.situations.append(rec)
        if len(self.situations) > self.max_situations * 1.2:
            self.situations = self.situations[-self.max_situations:]
            self._rebuild_index()

    # --- хранение ---
    def _load(self) -> None:
        d = json.loads(self.path.read_text(encoding="utf-8"))
        self.pending = d.get("pending", [])
        for k, v in d.get("stats", {}).items():
            self.stats[k] = v
        self.situations = d.get("situations", [])
        # старый формат уроков - просто строки
        self.lessons = [l if isinstance(l, dict) else {"text": l, "f": None, "ticker": ""}
                        for l in d.get("lessons", [])]
        self.decisions = d.get("decisions", {})
        self.processed = d.get("processed", [])

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if len(self.decisions) > 2000:
            self.decisions = dict(list(self.decisions.items())[-2000:])
        data = {"pending": self.pending[-5000:], "stats": self.stats,
                "situations": self.situations[-self.max_situations:], "lessons": self.lessons[-300:],
                "decisions": self.decisions, "processed": self.processed[-2000:]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    # --- точность агентов и база ситуаций ---
    def record_signals(self, ticker: str, price: float, signals: dict[str, float], time: str,
                       features: dict | None = None, interval=None) -> str:
        decision_id = uuid.uuid4().hex[:12]
        self.pending.append({"id": decision_id, "ticker": ticker, "price": price, "time": time,
                             "signals": signals, "age": 0, "f": features, "interval": interval})
        return decision_id

    def evaluate(self, ticker: str, price: float) -> None:
        """Вызывается каждый цикл: проверяет созревшие сигналы по текущей цене."""
        keep = []
        for rec in self.pending:
            if rec["ticker"] != ticker:
                keep.append(rec)
                continue
            rec["age"] += 1
            if rec["age"] < self.eval_after:
                keep.append(rec)
                continue
            ret = price / rec["price"] - 1
            if rec.get("f") is not None:
                self._add_situation({"f": rec["f"], "ret": round(ret, 6), "interval": rec.get("interval"),
                                     "ticker": ticker})
            if abs(ret) < 0.0005:   # цена почти не изменилась - точность не считаем
                continue
            for agent, sig in rec["signals"].items():
                if abs(sig) < 0.1 or agent not in self.stats:
                    continue   # нейтральный сигнал не оцениваем
                st = self.stats[agent]
                st["total"] += 1
                st["hits"] += int((sig > 0) == (ret > 0))
                st["acc"] = round(st["hits"] / st["total"], 3)
        self.pending = keep

    def weights(self) -> dict[str, float]:
        """Вес агента от 0.1 до 3. Пока данных мало - все равны (1.0).

        Важна не точность сама по себе, а преимущество над монеткой (50%): агент с точностью 70%
        получает вес 3, с точностью 50% - 1, с 45% и ниже - почти ноль. Чем больше проверок,
        тем смелее перераспределяем вес (при малой выборке 60% может быть случайностью)."""
        out = {}
        for a, st in self.stats.items():
            n = st["total"]
            if n < self.min_samples:
                out[a] = 1.0
                continue
            trust = min(1.0, n / 200)              # доверие к статистике растёт с числом проверок
            edge = (st["acc"] - 0.5) * 10          # 0.6 -> +1, 0.45 -> -0.5
            out[a] = round(float(np.clip(1 + edge * trust, 0.1, 3.0)), 3)
        return out

    def similar(self, features: dict, interval=None, k: int = 20, min_base: int = 50) -> dict:
        """Что было в k самых похожих прошлых ситуациях на том же таймфрейме."""
        n = len(self.situations)
        mask = np.ones(n, dtype=bool) if interval is None else np.array([x == interval for x in self._ints], bool)
        base = int(mask.sum())
        if base < min_base:
            return {"available": False, "base": base}
        mat, rets = self._mat[:n][mask], self._rets[:n][mask]
        dist = np.linalg.norm(mat - _vec(features), axis=1)
        idx = np.argpartition(dist, min(k, base - 1))[:k]
        r = rets[idx]
        return {
            "available": True, "base": base, "n": int(len(idx)),
            "up_share": round(float((r > 0).mean()), 3),
            "avg_return": round(float(r.mean()), 5),
            "median_return": round(float(np.median(r)), 5),
            "worst": round(float(r.min()), 5), "best": round(float(r.max()), 5),
            "similarity": round(float(1 / (1 + dist[idx].mean())), 3),
        }

    # --- уроки ---
    def remember_decision(self, decision_id: str, summary: dict) -> None:
        self.decisions[decision_id] = summary

    def relevant_lessons(self, features: dict | None, ticker: str, k: int = 5) -> list[str]:
        """Уроки из самых похожих ситуаций + последние уроки по этой бумаге."""
        if not self.lessons:
            return []
        out: list[str] = []
        with_f = [l for l in self.lessons if l.get("f")]
        if features and with_f:
            v = _vec(features)
            with_f.sort(key=lambda l: float(np.linalg.norm(_vec(l["f"]) - v)))
            out += [l["text"] for l in with_f[:k]]
        out += [l["text"] for l in self.lessons if l.get("ticker") == ticker][-k:]
        seen, uniq = set(), []
        for t in out:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq[: 2 * k]

    def learn_from_trades(self, closed_trades: list[dict], llm: LLM | None = None,
                          anonymous: bool = False) -> list[str]:
        """anonymous - в бэктесте не показываем Claude тикер и даты (чтобы он не «вспомнил» историю)."""
        hidden = {"ticker", "opened_at", "closed_at", "decision_id", "time"}
        new = []
        done = set(self.processed)
        for t in closed_trades:
            key = f"{t['ticker']}|{t['closed_at']}|{t.get('decision_id', '')}|{t['exit']}"
            if key in done:
                continue
            done.add(key)
            self.processed.append(key)
            d = self.decisions.get(t.get("decision_id", ""), {})
            text = None
            label = "Бумага X" if anonymous else t["ticker"]
            if llm:
                tt = {k: v for k, v in t.items() if k not in hidden} if anonymous else t
                dd = {k: v for k, v in d.items() if k not in hidden} if anonymous else d
                data = json.dumps({"сделка": tt, "решение_при_входе": dd}, ensure_ascii=False, default=str)
                out = llm.ask("Ты - наставник трейдера. Разбери закрытую сделку: что было сделано правильно или "
                              "неправильно при входе и выходе, и сформулируй один полезный урок.", data, Lesson)
                text = out.lesson if out else None
            if not text:
                result = "прибыль" if t["pnl"] > 0 else "убыток"
                sim = d.get("similar") or {}
                hint = f", похожие ситуации давали рост в {sim['up_share']:.0%} случаев" if sim.get("available") else ""
                text = (f"{label}: {result} {t['pnl_pct']:+.2%}, выход «{t['reason']}». "
                        f"Сигнал команды при входе {d.get('debate_signal', 'н/д')}{hint}.")
            self.lessons.append({"text": text, "f": d.get("features"), "ticker": t["ticker"],
                                 "pnl_pct": t.get("pnl_pct")})
            new.append(text)
        return new
