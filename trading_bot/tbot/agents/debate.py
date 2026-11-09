"""Агент 4: Исследователи «Бык и Медведь» - спорят о сделке, чтобы найти слабые места."""
from __future__ import annotations

from ..models import AgentReport, DebateResult
from .base import Agent, MarketContext, as_json, clamp


class BullBearDebate(Agent):
    name = "debate"
    title = "4. Бык и Медведь"
    role_prompt = (
        "Ты проводишь спор двух исследователей. Бык ищет самые сильные аргументы ЗА покупку, Медведь - самые "
        "сильные аргументы ПРОТИВ (за продажу или ожидание). Каждый должен опровергать аргументы другого, опираясь "
        "на отчёты аналитиков. Затем как беспристрастный судья реши, чьи аргументы сильнее, и выдай итоговый "
        "сигнал от -1 до +1. Учитывай, насколько точными аналитики были в прошлом (agent_stats)."
    )

    def run(self, ctx: MarketContext, reports: dict[str, AgentReport],
            weights: dict[str, float]) -> DebateResult:
        if self.llm:
            data = (f"Бумага: {ctx.ticker} ({ctx.name}), цена {ctx.price}\n"
                    f"Отчёты аналитиков:\n{as_json({k: v.model_dump() for k, v in reports.items()})}\n"
                    f"Историческая точность аналитиков:\n{as_json(ctx.agent_stats)}\n"
                    f"Текущая позиция: {as_json(ctx.position)}")
            out = self.llm.ask(self.role_prompt, data, DebateResult)
            if out:
                out.signal, out.confidence = clamp(out.signal), clamp(out.confidence, 0, 1)
                return out
        return self.rules(reports, weights)

    @staticmethod
    def rules(reports: dict[str, AgentReport], weights: dict[str, float]) -> DebateResult:
        """Взвешенное голосование: вес = (уверенность агента) x (его историческая точность)."""
        num = den = 0.0
        bull, bear = [], []
        for name, r in reports.items():
            w = r.confidence * weights.get(name, 1.0)
            num += r.signal * w
            den += w
            (bull if r.signal > 0 else bear).extend(r.key_points[:2])
        signal = clamp(num / den) if den else 0.0
        agree = [r.signal for r in reports.values() if abs(r.signal) > 0.1]
        same_side = agree and (all(s > 0 for s in agree) or all(s < 0 for s in agree))
        conf = min(1.0, abs(signal) * (1.0 if same_side else 0.6) + 0.1)
        winner = "bull" if signal > 0.1 else "bear" if signal < -0.1 else "draw"
        return DebateResult(bull_case="; ".join(bull) or "сильных аргументов нет",
                            bear_case="; ".join(bear) or "сильных аргументов нет",
                            winner=winner, signal=round(signal, 3), confidence=round(conf, 3))
