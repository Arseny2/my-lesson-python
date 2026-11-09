"""Главный цикл: данные -> 3 аналитика -> спор Быка и Медведя -> Трейдер -> Риск-менеджер -> заявка."""
from __future__ import annotations

import logging
import math

import pandas as pd

from .agents import (BullBearDebate, MarketContext, NewsAnalyst, RiskManager, SentimentAnalyst,
                     TechnicalAnalyst, Trader)
from .behavior import candle_behavior, trade_flow
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
        self.technical = TechnicalAnalyst(llm)
        self.news = NewsAnalyst(llm)
        self.sentiment = SentimentAnalyst(llm)
        self.debate = BullBearDebate(llm)
        self.trader = Trader(llm)
        self.risk = RiskManager(cfg.risk, llm)
        self.prices: dict[str, float] = {}
        self._day: str | None = None
        self._day_start_equity: float | None = None
        self._trades_today = 0

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

    # --- один цикл по одной бумаге ---
    def run_cycle(self, ticker: str, candles: pd.DataFrame, *, name: str = "", lot_size: int = 1,
                  trades: pd.DataFrame | None = None, news=None, market_news=None,
                  exec_price: float | None = None, time: str = "", trade: bool = True) -> CycleResult:
        """candles - свечи ДО текущего момента включительно (без заглядывания в будущее).
        exec_price - цена, по которой будет исполнена заявка (в бэктесте - открытие следующей свечи)."""
        df = candles if "rsi14" in candles.columns else add_all(candles)
        last = df.iloc[-1]
        price = float(exec_price if exec_price is not None else last["close"])
        time = time or str(df.index[-1])
        self.prices[ticker] = price
        self._roll_day(time)
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
        new_lessons = self.memory.learn_from_trades(getattr(self.broker, "closed_trades", []), self.llm)
        for l in new_lessons:
            log.debug("Урок: %s", l)

        pos = self.broker.position(ticker)
        portfolio = self.broker.portfolio(self.prices)
        ctx = MarketContext(
            ticker=ticker, name=name or ticker, price=price, lot_size=lot_size, candles=df,
            indicators=snapshot(df), behavior=candle_behavior(df), flow=trade_flow(trades),
            news=news or [], market_news=market_news or [],
            position=portfolio["positions"].get(ticker) if pos else None,
            portfolio={k: portfolio[k] for k in ("cash", "equity", "exposure")},
            lessons=self.memory.lessons, agent_stats=self.memory.stats,
        )

        # 2. три аналитика
        result.reports = {
            "technical": self.technical.analyze(ctx),
            "news": self.news.analyze(ctx),
            "sentiment": self.sentiment.analyze(ctx),
        }
        # 3. спор
        result.debate = self.debate.run(ctx, result.reports, self.memory.weights())
        # 4. трейдер
        result.decision = self.trader.decide(ctx, result.reports, result.debate)
        # 5. риск-менеджер
        result.verdict = self.risk.review(ctx, result.decision, result.debate,
                                          self.day_pnl_pct(), self._trades_today)

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
