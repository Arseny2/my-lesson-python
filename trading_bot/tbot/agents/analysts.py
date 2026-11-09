"""Агенты 1-3: технический аналитик, новостной аналитик, аналитик настроения и поведения трейдеров."""
from __future__ import annotations

import numpy as np

from ..models import AgentReport
from .base import Agent, MarketContext, as_json, clamp, recent_bars


class TechnicalAnalyst(Agent):
    name = "technical"
    title = "1. Технический аналитик"
    role_prompt = (
        "Ты - технический аналитик. Изучи свечи и индикаторы (EMA, SMA200, RSI, MACD, полосы Боллинджера, ATR). "
        "Определи тренд, уровни поддержки и сопротивления, перекупленность/перепроданность, силу импульса. "
        "Выдай сигнал от -1 (продавать) до +1 (покупать) и уверенность 0..1."
    )

    def analyze(self, ctx: MarketContext) -> AgentReport:
        if self.llm:
            data = (f"Бумага: {ctx.ticker} ({ctx.name}), цена {ctx.price}\n"
                    f"Индикаторы:\n{as_json(ctx.indicators)}\n"
                    f"Последние свечи (CSV):\n{recent_bars(ctx.candles, 40)}")
            out = self.llm.ask(self.role_prompt, data, AgentReport)
            if out:
                out.signal, out.confidence = clamp(out.signal), clamp(out.confidence, 0, 1)
                return out
        return self.rules(ctx)

    @staticmethod
    def rules(ctx: MarketContext) -> AgentReport:
        i = ctx.indicators
        score, points = 0.0, []
        if i.get("ema20") and i.get("ema50"):
            if i["ema20"] > i["ema50"]:
                score += 0.35; points.append("EMA20 выше EMA50 - восходящий тренд")
            else:
                score -= 0.35; points.append("EMA20 ниже EMA50 - нисходящий тренд")
        if i.get("sma200"):
            score += 0.15 if i["close"] > i["sma200"] else -0.15
        if i.get("macd_hist") is not None:
            score += 0.2 if i["macd_hist"] > 0 else -0.2
            points.append(f"гистограмма MACD {'положительная' if i['macd_hist'] > 0 else 'отрицательная'}")
        r = i.get("rsi14")
        if r is not None:
            if r > 70:
                score -= 0.3; points.append(f"RSI {r:.0f} - перекупленность")
            elif r < 30:
                score += 0.3; points.append(f"RSI {r:.0f} - перепроданность")
        if i.get("bb_high") and i["close"] > i["bb_high"]:
            score -= 0.1; points.append("цена выше верхней полосы Боллинджера")
        elif i.get("bb_low") and i["close"] < i["bb_low"]:
            score += 0.1; points.append("цена ниже нижней полосы Боллинджера")
        score = clamp(score)
        return AgentReport(signal=round(score, 3), confidence=round(min(1.0, 0.3 + abs(score) * 0.6), 3),
                           summary="Алгоритмический тех. анализ: " + ("покупка" if score > 0.15 else
                                                                      "продажа" if score < -0.15 else "нейтрально"),
                           key_points=points)


class NewsAnalyst(Agent):
    name = "news"
    title = "2. Новостной аналитик"
    role_prompt = (
        "Ты - аналитик новостей российского фондового рынка. Оцени, как новости о компании и общие новости "
        "(решения ЦБ по ставке, курс рубля, нефть, санкции, дивиденды, отчётность) повлияют на цену акции "
        "в ближайшие дни. Старые или нерелевантные новости игнорируй. Нет новостей - сигнал 0, уверенность низкая."
    )

    POSITIVE = ["рост", "рекорд", "дивиденд", "прибыль выросла", "повысил", "buyback", "выкуп", "снизил ставку",
                "превысил", "одобрил"]
    NEGATIVE = ["падение", "убыток", "санкци", "снизил прогноз", "штраф", "повысил ставку", "отказ",
                "дефолт", "снижение прибыли", "иск"]

    def analyze(self, ctx: MarketContext) -> AgentReport:
        if not ctx.news and not ctx.market_news:
            return AgentReport(signal=0.0, confidence=0.05, summary="Свежих новостей нет", key_points=[])
        if self.llm:
            fmt = lambda items: "\n".join(
                f"- [{n.published:%d.%m %H:%M}] {n.title}. {n.summary[:300]}" if n.published else f"- {n.title}"
                for n in items) or "нет"
            data = (f"Бумага: {ctx.ticker} ({ctx.name})\nНовости о компании:\n{fmt(ctx.news)}\n\n"
                    f"Общие новости рынка:\n{fmt(ctx.market_news)}")
            out = self.llm.ask(self.role_prompt, data, AgentReport)
            if out:
                out.signal, out.confidence = clamp(out.signal), clamp(out.confidence, 0, 1)
                return out
        return self.rules(ctx)

    def rules(self, ctx: MarketContext) -> AgentReport:
        score, points = 0, []
        for n in ctx.news:
            t = f"{n.title} {n.summary}".lower()
            p = sum(w in t for w in self.POSITIVE)
            m = sum(w in t for w in self.NEGATIVE)
            if p or m:
                score += p - m
                points.append(n.title)
        s = clamp(np.tanh(score / 3))
        return AgentReport(signal=round(float(s), 3), confidence=0.2 if points else 0.05,
                           summary=f"Подсчёт ключевых слов в {len(ctx.news)} новостях", key_points=points[:5])


class SentimentAnalyst(Agent):
    name = "sentiment"
    title = "3. Аналитик поведения трейдеров"
    role_prompt = (
        "Ты - аналитик поведения участников рынка. По ленте сделок (кто инициатор: покупатель B или продавец S, "
        "что делают крупные игроки) и по свечам (объём на росте и падении, всплески объёма, индекс страха и "
        "жадности) определи, кто сейчас контролирует цену: покупатели или продавцы, есть ли паника, FOMO, "
        "накопление или распределение крупными игроками. Учитывай, что толпа часто ошибается на экстремумах."
    )

    def analyze(self, ctx: MarketContext) -> AgentReport:
        if self.llm:
            data = (f"Бумага: {ctx.ticker}, цена {ctx.price}\nПоток сделок:\n{as_json(ctx.flow)}\n"
                    f"Поведение по свечам:\n{as_json(ctx.behavior)}")
            out = self.llm.ask(self.role_prompt, data, AgentReport)
            if out:
                out.signal, out.confidence = clamp(out.signal), clamp(out.confidence, 0, 1)
                return out
        return self.rules(ctx)

    @staticmethod
    def rules(ctx: MarketContext) -> AgentReport:
        b, f = ctx.behavior, ctx.flow
        score, points = 0.0, []
        uvs = b.get("up_volume_share", 0.5)
        score += (uvs - 0.5) * 1.2
        points.append(f"доля объёма на растущих свечах {uvs:.0%}")
        if f.get("available"):
            score += f["imbalance"] * 0.8 + f["big_trades_imbalance"] * 0.6
            points.append(f"дисбаланс покупок/продаж {f['imbalance']:+.2f}, крупные игроки {f['big_trades_imbalance']:+.2f}")
        fg = b.get("fear_greed", 50)
        if fg > 80:
            score -= 0.25; points.append(f"жадность {fg:.0f}/100 - риск разворота")
        elif fg < 20:
            score += 0.25; points.append(f"страх {fg:.0f}/100 - возможен отскок")
        score = clamp(score)
        return AgentReport(signal=round(score, 3), confidence=round(min(1.0, 0.25 + abs(score) * 0.5), 3),
                           summary="Алгоритмическая оценка поведения толпы и крупных игроков", key_points=points)
