"""Бэктест: прогон команды агентов по истории. Решение принимается на закрытии свечи,
исполняется по открытию следующей - так нет «подглядывания в будущее»."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .brokers import PaperBroker
from .config import Config
from .engine import TradingEngine
from .indicators import add_all
from .llm import LLM
from .memory import Memory


@dataclass
class BacktestReport:
    ticker: str
    bars: int
    start_equity: float
    end_equity: float
    total_return: float
    buy_hold_return: float
    max_drawdown: float
    sharpe: float
    trades: int
    win_rate: float
    agent_accuracy: dict
    lessons: list[str] = field(default_factory=list)
    equity_curve: pd.Series | None = None


def bars_per_year(interval) -> float:
    # ~252 торговых дня, основная сессия ~9 часов (плюс вечерняя - берём ~14 часов)
    if not isinstance(interval, int):
        interval = 60
    if interval >= 1440 or interval == 24:
        return 252
    return 252 * 14 * 60 / interval


def run_backtest(cfg: Config, ticker: str, candles: pd.DataFrame, lot_size: int = 1,
                 llm: LLM | None = None, warmup: int = 60, progress=None, window: int = 300,
                 start: int | None = None) -> BacktestReport:
    """start - с какой свечи начинать торговлю (до неё - только прогрев индикаторов).
    Удобно для проверки «вне выборки»: индикаторы считаются по всей истории, торгуем только на хвосте."""
    df = add_all(candles)  # все индикаторы причинные (смотрят только назад)
    intraday = cfg.flat_at_session_end and (isinstance(cfg.candle_interval, str) or cfg.candle_interval < 1440)
    dates = df.index.date
    broker = PaperBroker(cfg.start_cash, cfg.commission_pct, cfg.slippage_pct)
    memory = Memory()
    engine = TradingEngine(cfg, broker, memory, llm)
    equity = []
    n = len(df)
    if n <= warmup + 1:
        raise ValueError(f"Слишком мало свечей ({n}) - нужно больше {warmup + 1}")

    first = max(warmup, start or 0)
    for i in range(first, n - 1):
        # агентам хватает последних `window` свечей - так минутный бэктест не тормозит
        win = df.iloc[max(0, i + 1 - window): i + 1]
        next_open = float(df["open"].iloc[i + 1])
        t = str(df.index[i])
        if intraday and dates[i + 1] != dates[i]:
            # последняя свеча дня: новых сделок не открываем, позицию закрываем по закрытию
            engine.prices[ticker] = float(df["close"].iloc[i])
            broker.check_exits(ticker, {k: float(df[k].iloc[i]) for k in ("open", "high", "low", "close")}, t)
            engine.close_all(t, reason="конец дня")
        else:
            engine.run_cycle(ticker, win, lot_size=lot_size, exec_price=next_open, time=t)
        equity.append(broker.portfolio({ticker: float(df["close"].iloc[i + 1])})["equity"])
        if progress:
            progress(i - first + 1, n - 1 - first)

    curve = pd.Series(equity, index=df.index[first + 1:])
    rets = curve.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(bars_per_year(cfg.candle_interval))) if rets.std() else 0.0
    dd = float((curve / curve.cummax() - 1).min())
    closed = broker.closed_trades
    wins = sum(1 for t in closed if t["pnl"] > 0)
    return BacktestReport(
        ticker=ticker, bars=len(curve), start_equity=cfg.start_cash, end_equity=float(curve.iloc[-1]),
        total_return=float(curve.iloc[-1] / cfg.start_cash - 1),
        buy_hold_return=float(df["close"].iloc[-1] / df["open"].iloc[first + 1] - 1),
        max_drawdown=dd, sharpe=round(sharpe, 2), trades=len(closed),
        win_rate=wins / len(closed) if closed else 0.0,
        agent_accuracy={k: v for k, v in memory.stats.items()}, lessons=memory.lessons[-5:],
        equity_curve=curve,
    )
