"""Общая основа агентов: каждый умеет думать через Claude и имеет запасной алгоритм без ИИ."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from ..llm import LLM


@dataclass
class MarketContext:
    """Всё, что агенты знают о бумаге в данный момент."""

    ticker: str
    name: str
    price: float
    lot_size: int
    candles: pd.DataFrame                      # свечи с индикаторами
    indicators: dict
    behavior: dict
    flow: dict
    news: list = field(default_factory=list)   # list[NewsItem]
    market_news: list = field(default_factory=list)
    position: dict | None = None               # текущая позиция, если есть
    portfolio: dict = field(default_factory=dict)
    lessons: list[str] = field(default_factory=list)  # выводы из прошлых сделок
    agent_stats: dict = field(default_factory=dict)   # точность агентов в прошлом


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def as_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1, default=str)


def recent_bars(df: pd.DataFrame, n: int = 30) -> str:
    cols = ["open", "high", "low", "close", "volume"]
    tail = df[cols].tail(n).round(3)
    return tail.to_csv()


class Agent:
    name: str = "agent"
    title: str = "Агент"
    role_prompt: str = ""

    def __init__(self, llm: LLM | None = None):
        self.llm = llm
