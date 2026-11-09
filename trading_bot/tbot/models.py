"""Структуры данных, которыми обмениваются агенты."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class AgentReport(BaseModel):
    """Отчёт аналитика (технический, новостной, настроение рынка)."""

    signal: float = Field(description="Сигнал от -1 (сильно продавать) до +1 (сильно покупать)")
    confidence: float = Field(description="Уверенность от 0 до 1")
    summary: str = Field(description="Короткий вывод на русском, 1-3 предложения")
    key_points: list[str] = Field(default_factory=list, description="Главные факты, на которых основан вывод")


class DebateResult(BaseModel):
    """Итог спора быка и медведя."""

    bull_case: str = Field(description="Лучшие аргументы за покупку")
    bear_case: str = Field(description="Лучшие аргументы против покупки / за продажу")
    winner: Literal["bull", "bear", "draw"]
    signal: float = Field(description="Итоговый сигнал спора от -1 до +1")
    confidence: float = Field(description="Уверенность от 0 до 1")


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradeDecision(BaseModel):
    """Решение трейдера до проверки риск-менеджером."""

    action: Action
    size_pct: float = Field(description="Доля капитала на сделку, от 0 до 1 (для BUY); для SELL - доля позиции")
    stop_loss_pct: float = Field(description="Стоп-лосс в процентах от цены входа, например 0.03 = 3%")
    take_profit_pct: float = Field(description="Тейк-профит в процентах от цены входа")
    reasoning: str = Field(description="Почему принято такое решение, на русском")


class RiskVerdict(BaseModel):
    """Вердикт риск-менеджера."""

    approved: bool
    size_pct: float = Field(description="Разрешённая доля капитала (может быть меньше запрошенной)")
    stop_loss_pct: float
    take_profit_pct: float
    reason: str


@dataclass
class Position:
    ticker: str
    lots: int
    lot_size: int
    avg_price: float
    stop_price: float | None = None
    take_price: float | None = None
    opened_at: str = ""
    decision_id: str = ""

    @property
    def shares(self) -> int:
        return self.lots * self.lot_size


@dataclass
class Fill:
    ticker: str
    side: Action
    lots: int
    price: float
    commission: float
    time: str
    reason: str = ""


@dataclass
class CycleResult:
    """Всё, что произошло за один цикл анализа одного тикера."""

    ticker: str
    price: float
    reports: dict[str, AgentReport] = field(default_factory=dict)
    debate: DebateResult | None = None
    decision: TradeDecision | None = None
    verdict: RiskVerdict | None = None
    fill: Fill | None = None
    time: str = ""
    used_llm: bool = False
