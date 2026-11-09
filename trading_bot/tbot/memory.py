"""Память и самообучение бота.

Два механизма:
1. Точность агентов. Каждый сигнал аналитика через N циклов сверяется с тем, куда реально пошла цена.
   Агенты, которые чаще угадывают, получают больший вес в голосовании, те, кто ошибается, - меньший.
2. Уроки сделок. Когда позиция закрыта, бот записывает вывод (через Claude или по шаблону),
   и трейдер видит последние уроки перед каждым решением.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from .llm import LLM

ANALYSTS = ["technical", "news", "sentiment", "debate"]


class Lesson(BaseModel):
    lesson: str = Field(description="Один практический вывод на будущее, 1-2 предложения, по-русски")


class Memory:
    def __init__(self, path: str | Path | None = None, eval_after_cycles: int = 5, min_samples: int = 10):
        self.path = Path(path) if path else None
        self.eval_after = eval_after_cycles
        self.min_samples = min_samples
        self.pending: list[dict] = []           # сигналы, ждущие проверки
        self.stats: dict[str, dict] = {a: {"hits": 0, "total": 0, "acc": 0.5} for a in ANALYSTS}
        self.lessons: list[str] = []
        self.decisions: dict[str, dict] = {}    # decision_id -> краткая запись решения
        self.processed: list[str] = []         # ключи уже разобранных сделок
        if self.path and self.path.exists():
            self._load()

    # --- хранение ---
    def _load(self) -> None:
        d = json.loads(self.path.read_text(encoding="utf-8"))
        self.pending = d.get("pending", [])
        self.stats.update(d.get("stats", {}))
        self.lessons = d.get("lessons", [])
        self.decisions = d.get("decisions", {})
        self.processed = d.get("processed", [])

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # храним только последние решения, чтобы файл не рос бесконечно
        if len(self.decisions) > 1000:
            self.decisions = dict(list(self.decisions.items())[-1000:])
        data = {"pending": self.pending[-2000:], "stats": self.stats, "lessons": self.lessons[-50:],
                "decisions": self.decisions, "processed": self.processed[-1000:]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    # --- точность агентов ---
    def record_signals(self, ticker: str, price: float, signals: dict[str, float], time: str) -> str:
        decision_id = uuid.uuid4().hex[:12]
        self.pending.append({"id": decision_id, "ticker": ticker, "price": price, "time": time,
                             "signals": signals, "age": 0})
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
            if abs(ret) < 0.001:   # цена почти не изменилась - не считаем
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
        """Вес агента 0.5..1.5. Пока данных мало - все равны (1.0)."""
        out = {}
        for agent, st in self.stats.items():
            out[agent] = 1.0 if st["total"] < self.min_samples else round(0.5 + st["acc"], 3)
        return out

    # --- уроки ---
    def remember_decision(self, decision_id: str, summary: dict) -> None:
        self.decisions[decision_id] = summary

    def learn_from_trades(self, closed_trades: list[dict], llm: LLM | None = None) -> list[str]:
        new = []
        done = set(self.processed)
        for t in closed_trades:
            key = f"{t['ticker']}|{t['closed_at']}|{t.get('decision_id', '')}|{t['exit']}"
            if key in done:
                continue
            done.add(key)
            self.processed.append(key)
            d = self.decisions.get(t.get("decision_id", ""), {})
            lesson = None
            if llm:
                data = json.dumps({"сделка": t, "решение_при_входе": d}, ensure_ascii=False, default=str)
                out = llm.ask("Ты - наставник трейдера. Разбери закрытую сделку: что было сделано правильно или "
                              "неправильно при входе и выходе, и сформулируй один полезный урок.", data, Lesson)
                lesson = out.lesson if out else None
            if not lesson:
                result = "прибыль" if t["pnl"] > 0 else "убыток"
                lesson = (f"{t['ticker']}: {result} {t['pnl_pct']:+.1%}, выход по причине «{t['reason']}». "
                          f"Сигнал команды при входе: {d.get('debate_signal', 'н/д')}.")
            self.lessons.append(lesson)
            new.append(lesson)
        return new
