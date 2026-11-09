"""Агент 6: Риск-менеджер. Последнее слово за ним.

Жёсткие правила (лимиты из config.yaml) проверяются ВСЕГДА кодом, а не нейросетью:
ИИ может только ужесточить решение, но никогда не может обойти лимиты.
"""
from __future__ import annotations

from ..config import RiskConfig
from ..models import Action, DebateResult, RiskVerdict, TradeDecision
from .base import Agent, MarketContext, as_json


class RiskManager(Agent):
    name = "risk"
    title = "6. Риск-менеджер"
    role_prompt = (
        "Ты - риск-менеджер. Твоя задача - сохранить капитал. Проверь решение трейдера: не слишком ли большой "
        "размер, адекватен ли стоп-лосс волатильности (ATR), нет ли конфликта сигналов, не торгуем ли против "
        "сильного тренда, нет ли важных событий (отчётность, решение ЦБ). Ты можешь одобрить, уменьшить размер "
        "или запретить сделку. Жёсткие лимиты всё равно проверит код."
    )

    def __init__(self, cfg: RiskConfig, llm=None, round_trip_cost: float = 0.0):
        super().__init__(llm)
        self.cfg = cfg
        self.round_trip_cost = round_trip_cost

    def review(self, ctx: MarketContext, decision: TradeDecision, debate: DebateResult,
               day_pnl_pct: float, trades_today: int) -> RiskVerdict:
        verdict = None
        if self.llm and decision.action != Action.HOLD:
            data = (f"Бумага: {ctx.ticker}, цена {ctx.price}, ATR14 {ctx.indicators.get('atr14')}\n"
                    f"Решение трейдера:\n{as_json(decision.model_dump())}\n"
                    f"Спор:\n{as_json(debate.model_dump())}\nПортфель: {as_json(ctx.portfolio)}\n"
                    f"Лимиты: {as_json(self.cfg.__dict__)}\nPnL за день: {day_pnl_pct:.2%}")
            verdict = self.llm.ask(self.role_prompt, data, RiskVerdict)
        if verdict is None:
            verdict = RiskVerdict(approved=decision.action != Action.HOLD, size_pct=decision.size_pct,
                                  stop_loss_pct=decision.stop_loss_pct, take_profit_pct=decision.take_profit_pct,
                                  reason="Проверка по правилам")
        return self.enforce(ctx, decision, debate, verdict, day_pnl_pct, trades_today)

    def enforce(self, ctx: MarketContext, decision: TradeDecision, debate: DebateResult, verdict: RiskVerdict,
                day_pnl_pct: float, trades_today: int) -> RiskVerdict:
        """Жёсткие лимиты. ИИ не может их ослабить: берём минимум из его ответа и лимитов."""
        c = self.cfg
        reasons = [verdict.reason]

        def deny(why: str) -> RiskVerdict:
            return RiskVerdict(approved=False, size_pct=0.0, stop_loss_pct=verdict.stop_loss_pct,
                               take_profit_pct=verdict.take_profit_pct, reason=why)

        if decision.action == Action.HOLD:
            return deny("Трейдер решил не торговать")
        # Продажу имеющейся позиции (выход из риска) не блокируем никогда
        if decision.action == Action.SELL:
            if not ctx.position:
                return deny("Позиции нет, а шорт не поддерживается")
            return RiskVerdict(approved=True, size_pct=min(1.0, max(decision.size_pct, 0.0)) or 1.0,
                               stop_loss_pct=verdict.stop_loss_pct, take_profit_pct=verdict.take_profit_pct,
                               reason="Сокращение риска всегда разрешено. " + verdict.reason)

        # Дальше только BUY
        if not verdict.approved:
            return deny(verdict.reason)
        if day_pnl_pct <= -c.max_daily_loss_pct:
            return deny(f"Дневной убыток {day_pnl_pct:.2%} превысил лимит {c.max_daily_loss_pct:.0%} - стоп до завтра")
        if trades_today >= c.max_trades_per_day:
            return deny(f"Достигнут лимит сделок за день ({c.max_trades_per_day})")
        if debate.confidence < c.min_confidence:
            return deny(f"Уверенность команды {debate.confidence:.2f} ниже порога {c.min_confidence}")

        equity = ctx.portfolio.get("equity") or 0
        exposure = ctx.portfolio.get("exposure") or 0
        current_in_ticker = (ctx.position or {}).get("value", 0)
        room_ticker = max(0.0, c.max_position_pct - current_in_ticker / equity) if equity else 0
        room_total = max(0.0, c.max_total_exposure_pct - exposure / equity) if equity else 0
        atr = ctx.indicators.get("atr14") or 0
        min_stop = max(c.min_stop_loss_pct, c.atr_stop_multiplier * atr / ctx.price if ctx.price else 0)
        if min_stop > c.max_stop_loss_pct:
            return deny(f"Волатильность слишком высокая: нужный стоп {min_stop:.1%} больше лимита "
                        f"{c.max_stop_loss_pct:.0%}")
        stop = min(max(verdict.stop_loss_pct, min_stop), c.max_stop_loss_pct)
        take = max(verdict.take_profit_pct, stop * 1.5)  # прибыль минимум в 1.5 раза больше риска
        if stop != verdict.stop_loss_pct:
            reasons.append(f"стоп скорректирован до {stop:.2%}")
        need = self.round_trip_cost * c.min_edge_over_costs
        if take < need:
            return deny(f"Тейк {take:.2%} не окупает комиссии: нужно хотя бы {need:.2%} "
                        f"({c.min_edge_over_costs:g} x {self.round_trip_cost:.2%} за вход и выход)")

        # Размер от риска: если сработает стоп, теряем не больше risk_per_trade_pct капитала
        risk_size = c.risk_per_trade_pct / (stop + self.round_trip_cost)
        size = min(verdict.size_pct, decision.size_pct, room_ticker, room_total, risk_size)
        if size < verdict.size_pct:
            reasons.append(f"размер урезан до {size:.1%} по лимитам риска")
        lot_value = ctx.price * ctx.lot_size
        if equity and size * equity < lot_value:
            return deny(f"Разрешённой суммы ({size * equity:,.0f} ₽) не хватает даже на 1 лот ({lot_value:,.0f} ₽)")
        return RiskVerdict(approved=True, size_pct=round(size, 4), stop_loss_pct=round(stop, 4),
                           take_profit_pct=round(take, 4), reason="; ".join(reasons))
