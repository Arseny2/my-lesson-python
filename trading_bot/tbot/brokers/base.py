"""Общий интерфейс брокера."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Fill, Position


class Broker(ABC):
    name = "broker"
    is_real_money = False

    @abstractmethod
    def position(self, ticker: str) -> Position | None: ...

    @abstractmethod
    def portfolio(self, prices: dict[str, float]) -> dict:
        """{'cash', 'equity', 'exposure', 'positions': {ticker: {...}}}"""

    @abstractmethod
    def buy(self, ticker: str, lots: int, lot_size: int, price: float, stop: float | None,
            take: float | None, time: str, reason: str = "", decision_id: str = "") -> Fill | None: ...

    @abstractmethod
    def sell(self, ticker: str, lots: int, price: float, time: str, reason: str = "") -> Fill | None: ...

    def check_exits(self, ticker: str, bar: dict, time: str) -> Fill | None:
        """Сработали ли стоп-лосс или тейк-профит. bar = {'open','high','low','close'}."""
        pos = self.position(ticker)
        if not pos:
            return None
        if pos.stop_price and bar["low"] <= pos.stop_price:
            price = min(bar["open"], pos.stop_price)  # при гэпе вниз исполнимся по открытию
            return self.sell(ticker, pos.lots, price, time, reason="стоп-лосс")
        if pos.take_price and bar["high"] >= pos.take_price:
            price = max(bar["open"], pos.take_price)
            return self.sell(ticker, pos.lots, price, time, reason="тейк-профит")
        return None
