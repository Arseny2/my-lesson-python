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


def bars_per_year(interval: int) -> float:
    # ~252 торговых дня, основная сессия ~9 часов (плюс вечерняя - берём ~14 часов)
    if interval >= 1440 or interval == 24:
        return 252
    return 252 * 14 * 60 / interval


def run_backtest(cfg: Config, ticker: str, candles: pd.DataFrame, lot_size: int = 1,
                 llm: LLM | None = None, warmup: int = 60, progress=None) -> BacktestReport:
    df = add_all(candles)  # все индикаторы причинные (смотрят только назад)
    broker = PaperBroker(cfg.start_cash, cfg.commission_pct, cfg.slippage_pct)
    memory = Memory()
    engine = TradingEngine(cfg, broker, memory, llm)
    equity = []
    n = len(df)
    if n <= warmup + 1:
        raise ValueError(f"Слишком мало свечей ({n}) - нужно больше {warmup + 1}")

    for i in range(warmup, n - 1):
        window = df.iloc[: i + 1]
        next_open = float(df["open"].iloc[i + 1])
        engine.run_cycle(ticker, window, lot_size=lot_size, exec_price=next_open, time=str(df.index[i]))
        equity.append(broker.portfolio({ticker: float(df["close"].iloc[i + 1])})["equity"])
        if progress:
            progress(i - warmup + 1, n - 1 - warmup)

    curve = pd.Series(equity, index=df.index[warmup + 1:])
    rets = curve.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(bars_per_year(cfg.candle_interval))) if rets.std() else 0.0
    dd = float((curve / curve.cummax() - 1).min())
    closed = broker.closed_trades
    wins = sum(1 for t in closed if t["pnl"] > 0)
    return BacktestReport(
        ticker=ticker, bars=len(curve), start_equity=cfg.start_cash, end_equity=float(curve.iloc[-1]),
        total_return=float(curve.iloc[-1] / cfg.start_cash - 1),
        buy_hold_return=float(df["close"].iloc[-1] / df["open"].iloc[warmup + 1] - 1),
        max_drawdown=dd, sharpe=round(sharpe, 2), trades=len(closed),
        win_rate=wins / len(closed) if closed else 0.0,
        agent_accuracy={k: v for k, v in memory.stats.items()}, lessons=memory.lessons[-5:],
        equity_curve=curve,
    )
