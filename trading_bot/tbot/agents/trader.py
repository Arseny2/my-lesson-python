"""Агент 5: Трейдер - принимает решение BUY / SELL / HOLD и размер сделки."""
from __future__ import annotations

from ..models import Action, AgentReport, DebateResult, TradeDecision
from .base import Agent, MarketContext, as_json


class Trader(Agent):
    name = "trader"
    title = "5. Трейдер"
    role_prompt = (
        "Ты - внутридневной трейдер. На основе отчётов аналитиков и итогов спора Быка и Медведя прими решение: BUY (купить или "
        "докупить), SELL (продать имеющуюся позицию) или HOLD (ничего не делать). Если позиции нет, SELL означает "
        "HOLD - шорт запрещён. Укажи долю капитала (size_pct, 0..1), стоп-лосс и тейк-профит в долях от цены "
        "(учитывай ATR - стоп не должен быть внутри обычного шума). Учитывай уроки прошлых сделок. "
        "Помни про комиссию: вход и выход вместе стоят round_trip_cost - тейк-профит должен быть заметно "
        "больше, иначе сделка убыточна даже при угаданном направлении. "
        "Не торгуй ради торговли: если сигнал слабый - HOLD."
    )

    def __init__(self, llm=None, threshold: float = 0.25, round_trip_cost: float = 0.0):
        super().__init__(llm)
        self.threshold = threshold
        self.round_trip_cost = round_trip_cost

    def decide(self, ctx: MarketContext, reports: dict[str, AgentReport], debate: DebateResult) -> TradeDecision:
        if self.llm:
            data = (f"Бумага: {ctx.label}, цена {ctx.price}, лот {ctx.lot_size} шт., "
                    f"ATR14 {ctx.indicators.get('atr14')}, round_trip_cost {self.round_trip_cost:.4f}\n"
                    f"Отчёты:\n{as_json({k: v.model_dump() for k, v in reports.items()})}\n"
                    f"Спор:\n{as_json(debate.model_dump())}\n"
                    f"Похожие ситуации в прошлом:\n{as_json(ctx.similar)}\n"
                    f"Позиция: {as_json(ctx.position)}\nПортфель: {as_json(ctx.portfolio)}\n"
                    f"Уроки из похожих прошлых сделок:\n" + ("\n".join(f"- {l}" for l in ctx.lessons[-10:]) or "пока нет"))
            out = self.llm.ask(self.role_prompt, data, TradeDecision)
            if out:
                return out
        return self.rules(ctx, debate)

    def rules(self, ctx: MarketContext, debate: DebateResult) -> TradeDecision:
        atr = ctx.indicators.get("atr14") or ctx.price * 0.01
        stop = max(0.002, min(0.05, 2 * atr / ctx.price))
        # тейк: минимум 2 стопа и минимум 3 комиссии за круг
        take = max(stop * 2, self.round_trip_cost * 3)
        s, c = debate.signal, debate.confidence
        if s > self.threshold and not ctx.position:
            return TradeDecision(action=Action.BUY, size_pct=round(min(1.0, 0.5 + 0.5 * s * c), 3),
                                 stop_loss_pct=round(stop, 4), take_profit_pct=round(take, 4),
                                 reasoning=f"Сигнал команды {s:+.2f} при уверенности {c:.2f} - покупаем")
        if s < -self.threshold and ctx.position:
            return TradeDecision(action=Action.SELL, size_pct=1.0, stop_loss_pct=round(stop, 4),
                                 take_profit_pct=round(take, 4),
                                 reasoning=f"Сигнал команды {s:+.2f} - закрываем позицию")
        return TradeDecision(action=Action.HOLD, size_pct=0.0, stop_loss_pct=round(stop, 4),
                             take_profit_pct=round(take, 4),
                             reasoning=f"Сигнал {s:+.2f} недостаточно сильный или позиция уже соответствует ему")
