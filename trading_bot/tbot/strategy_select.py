"""Автовыбор: какие акции торговать и на каком таймфрейме (минутки, 10 минут, час).

1. Отбор бумаг: ликвидные, торгуются через API, и хотя бы 1 лот помещается в капитал.
2. Для каждой бумаги и каждого таймфрейма - бэктест с комиссиями.
   История делится на две части: на первых 70% подбираются параметры (обучение),
   на последних 30% - честная проверка на данных, которых бот «не видел».
3. Выигрывает вариант, который заработал на обеих частях. Если такого нет - бот честно
   говорит, что прибыльной стратегии не нашёл, и реальную торговлю не советует.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

from .backtest import run_backtest
from .config import Config

log = logging.getLogger(__name__)

# сколько дней истории брать для каждого таймфрейма
HISTORY_DAYS = {1: 10, 2: 10, 3: 10, 5: 20, 10: 30, 15: 30, 30: 60, 60: 120, 1440: 730}
THRESHOLDS = [0.2, 0.3, 0.4]


def screen_tickers(cfg: Config, data, log_fn=print) -> list[dict]:
    """Ликвидные бумаги, у которых лот по карману."""
    budget = cfg.start_cash * min(1.0, cfg.risk.max_position_pct)
    found = []
    for t in cfg.candidates:
        try:
            info = data.info(t)
        except Exception as e:
            log_fn(f"  {t}: пропуск ({e})")
            continue
        if info.get("tradable") is False or not info.get("last"):
            continue
        lot_value = info["lot_size"] * info["last"]
        if lot_value > budget:
            log_fn(f"  {t}: лот {lot_value:,.0f} ₽ дороже бюджета {budget:,.0f} ₽")
            continue
        try:
            daily = data.candles(t, 60, 5)
            turnover = float(daily["value"].sum()) / 5
        except Exception:
            turnover = 0.0
        found.append({**info, "lot_value": lot_value, "turnover": turnover})
        log_fn(f"  {t}: лот {lot_value:,.0f} ₽, оборот ~{turnover / 1e6:,.0f} млн ₽/день")
    found.sort(key=lambda x: x["turnover"], reverse=True)
    return found


def evaluate(cfg: Config, ticker: str, candles, lot_size: int, interval: int) -> list[dict]:
    """Перебор порога входа на обучающей части + проверка лучшего на тестовой."""
    split = int(len(candles) * 0.7)
    rows = []
    for th in THRESHOLDS:
        c = dataclasses.replace(cfg, candle_interval=interval, entry_threshold=th)
        train = run_backtest(c, ticker, candles.iloc[:split], lot_size=lot_size)
        rows.append({"threshold": th, "train": train})
    best = max(rows, key=lambda r: (r["train"].trades >= 3, r["train"].total_return))
    c = dataclasses.replace(cfg, candle_interval=interval, entry_threshold=best["threshold"])
    test = run_backtest(c, ticker, candles, lot_size=lot_size, start=split)  # прогрев на обучающей части
    tr = best["train"]
    return [{
        "ticker": ticker, "interval": interval, "threshold": best["threshold"],
        "train_return": round(tr.total_return, 4), "train_trades": tr.trades,
        "test_return": round(test.total_return, 4), "test_trades": test.trades,
        "test_win_rate": round(test.win_rate, 3), "test_max_dd": round(test.max_drawdown, 4),
        "test_buy_hold": round(test.buy_hold_return, 4), "lot_size": lot_size,
    }]


def select_strategy(cfg: Config, data, history, log_fn=print, max_tickers: int | None = None) -> dict:
    """data - источник текущих цен/лотов, history - источник исторических свечей (обычно MOEX ISS)."""
    log_fn("Шаг 1. Отбор бумаг по цене лота и ликвидности")
    tickers = cfg.candidates if cfg.tickers == "auto" else cfg.tickers
    cfg_screen = dataclasses.replace(cfg, candidates=list(tickers))
    pool = screen_tickers(cfg_screen, data, log_fn)[: max_tickers or cfg.max_tickers * 2]
    if not pool:
        return {"profitable": False, "tickers": [], "reason": "Нет бумаг, лот которых помещается в капитал",
                "results": []}

    log_fn("Шаг 2. Бэктест каждого таймфрейма с комиссией "
           f"{cfg.commission_pct:.2%} + проскальзывание {cfg.slippage_pct:.2%}")
    results = []
    for p in pool:
        for interval in cfg.test_intervals:
            try:
                candles = history.candles(p["ticker"], interval, HISTORY_DAYS.get(interval, 30))
            except Exception as e:
                log_fn(f"  {p['ticker']} {interval} мин: нет истории ({e})")
                continue
            if len(candles) < 300:
                log_fn(f"  {p['ticker']} {interval} мин: мало свечей ({len(candles)})")
                continue
            for r in evaluate(cfg, p["ticker"], candles, p["lot_size"], interval):
                results.append(r)
                log_fn(f"  {r['ticker']:6s} {interval:>4} мин  порог {r['threshold']:.1f}  "
                       f"обучение {r['train_return']:+.2%} ({r['train_trades']} сд.)  "
                       f"проверка {r['test_return']:+.2%} ({r['test_trades']} сд.)  "
                       f"купить-и-держать {r['test_buy_hold']:+.2%}")

    good = [r for r in results if r["train_return"] > 0 and r["test_return"] > 0 and r["test_trades"] >= 2]
    good.sort(key=lambda r: r["test_return"], reverse=True)
    if not good:
        return {"profitable": False, "tickers": [], "results": results,
                "reason": "Ни один вариант не заработал и на обучении, и на проверке с учётом комиссий"}

    # один таймфрейм на всю сессию - тот, где лучше всего в сумме по прошедшим бумагам
    by_interval: dict[int, float] = {}
    for r in good:
        by_interval[r["interval"]] = by_interval.get(r["interval"], 0) + r["test_return"]
    interval = max(by_interval, key=by_interval.get)
    chosen = [r for r in good if r["interval"] == interval][: cfg.max_tickers]
    return {
        "profitable": True,
        "interval": interval,
        "tickers": [r["ticker"] for r in chosen],
        "thresholds": {r["ticker"]: r["threshold"] for r in chosen},
        "results": results,
        "commission_pct": cfg.commission_pct,
    }


def save_strategy(path: Path, strategy: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(strategy, ensure_ascii=False, indent=1), encoding="utf-8")


def load_strategy(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
