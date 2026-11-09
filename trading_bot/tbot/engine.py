"""Главный цикл: данные -> 3 аналитика -> спор Быка и Медведя -> Трейдер -> Риск-менеджер -> заявка.

Режимы работы Claude (llm_mode):
  * always   - все агенты думают через Claude в каждом цикле (дорого и медленно на минутках);
  * escalate - сначала бесплатный алгоритмический прогон всех 6 агентов. Claude подключается,
               только если команда видит возможную сделку (сильный сигнал или открытая позиция),
               и пока не исчерпан бюджет. Так бюджет тратится на важные моменты, а не на шум.
"""
from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .agents import (BullBearDebate, MarketContext, NewsAnalyst, RiskManager, SentimentAnalyst,
                     TechnicalAnalyst, Trader)
from .behavior import book_pressure, candle_behavior, trade_flow
from .brokers import Broker
from .config import Config
from .indicators import add_all, snapshot
from .llm import LLM
from .memory import Memory
from .models import Action, CycleResult

log = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, cfg: Config, broker: Broker, memory: Memory, llm: LLM | None = None):
        self.cfg = cfg
        self.broker = broker
        self.memory = memory
        self.llm = llm
        cost = cfg.round_trip_cost_pct()
        self.technical = TechnicalAnalyst()
        self.news = NewsAnalyst()
        self.sentiment = SentimentAnalyst()
        self.debate = BullBearDebate()
        self.trader = Trader(threshold=cfg.entry_threshold, round_trip_cost=cost)
        self.risk = RiskManager(cfg.risk, round_trip_cost=cost)
        self.agents = [self.technical, self.news, self.sentiment, self.debate, self.trader, self.risk]
        self.prices: dict[str, float] = {}
        self._day: str | None = None
        self._day_start_equity: float | None = None
        self._trades_today = 0
        self.llm_cycles = 0

    def _set_llm(self, on: bool) -> None:
        for a in self.agents:
            a.llm = self.llm if on else None

    # --- дневные лимиты ---
    def _roll_day(self, time: str) -> None:
        day = str(time)[:10]
        if day != self._day:
            self._day = day
            self._day_start_equity = self.broker.portfolio(self.prices)["equity"]
            self._trades_today = 0

    def day_pnl_pct(self) -> float:
        if not self._day_start_equity:
            return 0.0
        return self.broker.portfolio(self.prices)["equity"] / self._day_start_equity - 1

    # --- прогон команды агентов ---
    def _think(self, ctx: MarketContext, result: CycleResult, use_llm: bool) -> None:
        self._set_llm(use_llm)
        try:
            if use_llm:
                # три аналитика независимы - спрашиваем Claude параллельно, чтобы уложиться в минуту
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futs = {k: pool.submit(a.analyze, ctx) for k, a in
                            (("technical", self.technical), ("news", self.news), ("sentiment", self.sentiment))}
                    result.reports = {k: f.result() for k, f in futs.items()}
            else:
                result.reports = {"technical": self.technical.analyze(ctx), "news": self.news.analyze(ctx),
                                  "sentiment": self.sentiment.analyze(ctx)}
            result.debate = self.debate.run(ctx, result.reports, self.memory.weights())
            result.decision = self.trader.decide(ctx, result.reports, result.debate)
            result.verdict = self.risk.review(ctx, result.decision, result.debate,
                                              self.day_pnl_pct(), self._trades_today)
        finally:
            self._set_llm(False)

    def _worth_escalating(self, ctx: MarketContext, result: CycleResult) -> bool:
        if not self.llm or not self.llm.can_spend():
            return False
        if self.cfg.llm_mode == "always":
            return True
        if ctx.position:            # есть позиция - важно вовремя выйти
            return True
        return abs(result.debate.signal) >= self.cfg.escalate_threshold

    # --- один цикл по одной бумаге ---
    def run_cycle(self, ticker: str, candles: pd.DataFrame, *, name: str = "", lot_size: int = 1,
                  trades: pd.DataFrame | None = None, book: dict | None = None, news=None, market_news=None,
                  exec_price: float | None = None, time: str = "", trade: bool = True,
                  entry_threshold: float | None = None) -> CycleResult:
        """candles - свечи ДО текущего момента включительно (без заглядывания в будущее).
        exec_price - цена, по которой будет исполнена заявка (в бэктесте - открытие следующей свечи)."""
        df = candles if "rsi14" in candles.columns else add_all(candles)
        last = df.iloc[-1]
        price = float(exec_price if exec_price is not None else last["close"])
        time = time or str(df.index[-1])
        self.prices[ticker] = price
        self._roll_day(time)
        self.trader.threshold = entry_threshold if entry_threshold is not None else self.cfg.entry_threshold
        result = CycleResult(ticker=ticker, price=price, time=time)

        # 0. сначала проверяем стопы/тейки по последней свече
        if trade:
            bar = {k: float(last[k]) for k in ("open", "high", "low", "close")}
            exit_fill = self.broker.check_exits(ticker, bar, time)
            if exit_fill:
                result.fill = exit_fill
                self._trades_today += 1

        # 1. самообучение: проверяем, кто из агентов был прав раньше
        self.memory.evaluate(ticker, price)
        for lesson in self.memory.learn_from_trades(getattr(self.broker, "closed_trades", []), self.llm):
            log.debug("Урок: %s", lesson)

        pos = self.broker.position(ticker)
        portfolio = self.broker.portfolio(self.prices)
        ctx = MarketContext(
            ticker=ticker, name=name or ticker, price=price, lot_size=lot_size, candles=df,
            indicators=snapshot(df), behavior=candle_behavior(df), flow=trade_flow(trades),
            book=book_pressure(book), news=news or [], market_news=market_news or [],
            position=portfolio["positions"].get(ticker) if pos else None,
            portfolio={k: portfolio[k] for k in ("cash", "equity", "exposure")},
            lessons=self.memory.lessons, agent_stats=self.memory.stats,
        )

        # 2-5. бесплатный прогон, затем (если стоит того) прогон с Claude
        self._think(ctx, result, use_llm=False)
        if self._worth_escalating(ctx, result):
            self._think(ctx, result, use_llm=True)
            result.used_llm = True
            self.llm_cycles += 1

        signals = {k: r.signal for k, r in result.reports.items()}
        signals["debate"] = result.debate.signal
        decision_id = self.memory.record_signals(ticker, price, signals, time)

        # 6. исполнение
        if trade and result.verdict.approved and result.fill is None:
            fill = self._execute(ctx, result, decision_id)
            if fill:
                result.fill = fill
                self._trades_today += 1
                self.memory.remember_decision(decision_id, {
                    "ticker": ticker, "time": time, "price": price, "action": fill.side.value,
                    "debate_signal": result.debate.signal, "reasoning": result.decision.reasoning,
                    "signals": signals,
                })
        self.memory.save()
        return result

    def _execute(self, ctx: MarketContext, result: CycleResult, decision_id: str):
        v, d = result.verdict, result.decision
        if d.action == Action.BUY:
            equity = ctx.portfolio["equity"]
            lots = math.floor(v.size_pct * equity / (ctx.price * ctx.lot_size))
            if lots <= 0:
                return None
            return self.broker.buy(ctx.ticker, lots, ctx.lot_size, ctx.price,
                                   stop=round(ctx.price * (1 - v.stop_loss_pct), 4),
                                   take=round(ctx.price * (1 + v.take_profit_pct), 4),
                                   time=result.time, reason=d.reasoning[:200], decision_id=decision_id)
        if d.action == Action.SELL:
            pos = self.broker.position(ctx.ticker)
            if not pos:
                return None
            lots = max(1, math.ceil(pos.lots * min(1.0, v.size_pct)))
            return self.broker.sell(ctx.ticker, lots, ctx.price, result.time, reason=d.reasoning[:200])
        return None

    def close_all(self, time: str, reason: str = "конец сессии") -> list:
        """Закрыть все позиции (вызывается в конце торговой сессии)."""
        fills = []
        for ticker in list(self.broker.portfolio(self.prices)["positions"]):
            pos = self.broker.position(ticker)
            if pos:
                f = self.broker.sell(ticker, pos.lots, self.prices.get(ticker, pos.avg_price), time, reason=reason)
                if f:
                    fills.append(f)
        return fills
