"""Агент 4: Исследователи «Бык и Медведь» + судья.

debate_rounds = 1: быстрый режим (минутки) - Claude разыгрывает спор и выносит решение за ОДИН запрос.
debate_rounds >= 2: настоящий спор, как в TradingAgents: Бык и Медведь - отдельные запросы, каждый
отвечает на аргументы соперника, затем судья (менеджер исследований) читает стенограмму и решает.
Стоит 2 x раунды + 1 запросов - подходит для 10-минутных и часовых свечей.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import AgentReport, DebateResult
from .base import Agent, MarketContext, as_json, clamp

DATA_RULES = ("Опирайся на отчёты аналитиков, память похожих ситуаций и точность аналитиков в прошлом. "
              "Не выдумывай факты.")


class Argument(BaseModel):
    argument: str = Field(description="Аргументы и ответ на доводы соперника, 2-5 предложений")
    strength: float = Field(description="Насколько сильна твоя позиция по-честному, 0..1")


class BullBearDebate(Agent):
    name = "debate"
    title = "4. Бык и Медведь"
    role_prompt = (
        "Ты проводишь спор двух исследователей. Бык ищет самые сильные аргументы ЗА покупку, Медведь - самые "
        "сильные аргументы ПРОТИВ (за продажу или ожидание). Каждый должен опровергать аргументы другого. "
        "Затем как беспристрастный судья реши, чьи аргументы сильнее, и выдай итоговый сигнал от -1 до +1. "
        + DATA_RULES
    )
    bull_prompt = ("Ты - исследователь-Бык. Найди самые сильные аргументы ЗА покупку и опровергни доводы Медведя, "
                   "если они есть. Будь честен: не преувеличивай. " + DATA_RULES)
    bear_prompt = ("Ты - исследователь-Медведь. Найди самые сильные аргументы ПРОТИВ покупки (риски, слабость, "
                   "перекупленность) и опровергни доводы Быка. Будь честен: не преувеличивай. " + DATA_RULES)
    judge_prompt = ("Ты - менеджер исследований и беспристрастный судья. Прочитай стенограмму спора Быка и "
                    "Медведя и данные. Реши, чьи аргументы подкреплены фактами лучше, и выдай итоговый сигнал "
                    "от -1 до +1 и уверенность. Если аргументы равны - сигнал около 0. " + DATA_RULES)

    def __init__(self, llm=None, rounds: int = 1):
        super().__init__(llm)
        self.rounds = rounds

    def _data(self, ctx: MarketContext, reports: dict[str, AgentReport]) -> str:
        return (f"Бумага: {ctx.label}, цена {ctx.price}\n"
                f"Отчёты аналитиков:\n{as_json({k: v.model_dump() for k, v in reports.items()})}\n"
                f"Похожие ситуации в прошлом:\n{as_json(ctx.similar)}\n"
                f"Историческая точность аналитиков:\n{as_json(ctx.agent_stats)}\n"
                f"Текущая позиция: {as_json(ctx.position)}")

    def run(self, ctx: MarketContext, reports: dict[str, AgentReport],
            weights: dict[str, float]) -> DebateResult:
        if self.llm:
            out = self._multi_round(ctx, reports) if self.rounds >= 2 else \
                self.llm.ask(self.role_prompt, self._data(ctx, reports), DebateResult)
            if out:
                out.signal, out.confidence = clamp(out.signal), clamp(out.confidence, 0, 1)
                return out
        return self.rules(reports, weights)

    def _multi_round(self, ctx: MarketContext, reports: dict[str, AgentReport]) -> DebateResult | None:
        data = self._data(ctx, reports)
        transcript: list[str] = []
        for rnd in range(1, self.rounds + 1):
            for side, prompt, label in (("bull", self.bull_prompt, "Бык"), ("bear", self.bear_prompt, "Медведь")):
                history = "\n".join(transcript) or "спор только начинается"
                arg = self.llm.ask(prompt, f"{data}\n\nСтенограмма спора:\n{history}", Argument)
                if arg is None:
                    return None   # бюджет кончился или API недоступен - решит алгоритм
                transcript.append(f"[Раунд {rnd}] {label} (сила {arg.strength:.2f}): {arg.argument}")
        return self.llm.ask(self.judge_prompt, f"{data}\n\nСтенограмма спора:\n" + "\n".join(transcript),
                            DebateResult)

    @staticmethod
    def rules(reports: dict[str, AgentReport], weights: dict[str, float]) -> DebateResult:
        """Взвешенное голосование: вес = (уверенность агента) x (его историческая точность)."""
        num = den = 0.0
        bull, bear = [], []
        for name, r in reports.items():
            w = r.confidence * weights.get(name, 1.0)
            num += r.signal * w
            den += w
            if abs(r.signal) > 0.05:
                (bull if r.signal > 0 else bear).extend(r.key_points[:2] or [r.summary])
        signal = clamp(num / den) if den else 0.0
        agree = [r.signal for r in reports.values() if abs(r.signal) > 0.1]
        same_side = agree and (all(s > 0 for s in agree) or all(s < 0 for s in agree))
        conf = min(1.0, abs(signal) * (1.0 if same_side else 0.6) + 0.1)
        winner = "bull" if signal > 0.1 else "bear" if signal < -0.1 else "draw"
        return DebateResult(bull_case="; ".join(bull) or "сильных аргументов нет",
                            bear_case="; ".join(bear) or "сильных аргументов нет",
                            winner=winner, signal=round(signal, 3), confidence=round(conf, 3))
