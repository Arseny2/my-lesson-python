"""Агент 5: Трейдер - принимает решение BUY / SELL / HOLD и размер сделки."""
from __future__ import annotations

from ..models import Action, AgentReport, DebateResult, TradeDecision
from .base import Agent, MarketContext, as_json


class Trader(Agent):
    name = "trader"
    title = "5. Трейдер"
    role_prompt = (
        "Ты - трейдер. На основе отчётов аналитиков и итогов спора Быка и Медведя прими решение: BUY (купить или "
        "докупить), SELL (продать имеющуюся позицию) или HOLD (ничего не делать). Если позиции нет, SELL означает "
        "HOLD - шорт запрещён. Укажи долю капитала (size_pct, 0..0.25), стоп-лосс и тейк-профит в долях от цены "
        "(учитывай ATR - стоп не должен быть внутри обычного шума). Учитывай уроки прошлых сделок. "
        "Не торгуй ради торговли: если сигнал слабый - HOLD."
    )

    def decide(self, ctx: MarketContext, reports: dict[str, AgentReport], debate: DebateResult) -> TradeDecision:
        if self.llm:
            data = (f"Бумага: {ctx.ticker} ({ctx.name}), цена {ctx.price}, лот {ctx.lot_size} шт., "
                    f"ATR14 {ctx.indicators.get('atr14')}\n"
                    f"Отчёты:\n{as_json({k: v.model_dump() for k, v in reports.items()})}\n"
                    f"Спор:\n{as_json(debate.model_dump())}\n"
                    f"Позиция: {as_json(ctx.position)}\nПортфель: {as_json(ctx.portfolio)}\n"
                    f"Уроки прошлых сделок:\n" + ("\n".join(f"- {l}" for l in ctx.lessons[-10:]) or "пока нет"))
            out = self.llm.ask(self.role_prompt, data, TradeDecision)
            if out:
                return out
        return self.rules(ctx, debate)

    @staticmethod
    def rules(ctx: MarketContext, debate: DebateResult, threshold: float = 0.2) -> TradeDecision:
        atr = ctx.indicators.get("atr14") or ctx.price * 0.02
        stop = max(0.01, min(0.08, 2 * atr / ctx.price))
        s, c = debate.signal, debate.confidence
        if s > threshold and not ctx.position:
            return TradeDecision(action=Action.BUY, size_pct=round(min(0.25, 0.1 + 0.15 * s * c), 3),
                                 stop_loss_pct=round(stop, 4), take_profit_pct=round(stop * 2, 4),
                                 reasoning=f"Сигнал команды {s:+.2f} при уверенности {c:.2f} - покупаем")
        if s < -threshold and ctx.position:
            return TradeDecision(action=Action.SELL, size_pct=1.0, stop_loss_pct=round(stop, 4),
                                 take_profit_pct=round(stop * 2, 4),
                                 reasoning=f"Сигнал команды {s:+.2f} - закрываем позицию")
        return TradeDecision(action=Action.HOLD, size_pct=0.0, stop_loss_pct=round(stop, 4),
                             take_profit_pct=round(stop * 2, 4),
                             reasoning=f"Сигнал {s:+.2f} недостаточно сильный или позиция уже соответствует ему")
