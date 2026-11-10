"""Обёртка над Claude API: один вызов = один ответ строго по схеме (pydantic)."""
from __future__ import annotations

import logging
import threading
import time
from typing import TypeVar

import anthropic
from pydantic import BaseModel

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

# Цены за 1M токенов (вход/выход) - для подсчёта расходов в терминале
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}

COMMON_RULES = (
    "Ты - участник команды ИИ-агентов, которая торгует российскими акциями на Мосбирже. "
    "Отвечай только на основе переданных данных, не выдумывай цифры и новости. "
    "Если данных мало или они противоречат друг другу - снижай уверенность, а не угадывай. "
    "Все тексты пиши по-русски, коротко и по делу."
)


class LLM:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium",
                 budget_usd: float | None = None, session_hours: float = 6.0):
        # Ключ берётся из ANTHROPIC_API_KEY (или профиля `ant auth login`)
        self.client = anthropic.Anthropic(max_retries=2, timeout=60.0)
        self.model = model
        self.effort = effort
        self.budget_usd = budget_usd
        self.session_seconds = session_hours * 3600
        self.started = time.monotonic()
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.errors = 0
        self.skipped = 0   # запросы, пропущенные из-за бюджета
        self._lock = threading.Lock()  # агенты могут спрашивать Claude параллельно

    def budget_left(self) -> float | None:
        return None if self.budget_usd is None else max(0.0, self.budget_usd - self.cost_usd)

    def can_spend(self) -> bool:
        """Тратим бюджет равномерно: к середине сессии - не больше ~половины (+10% запаса на всплеск)."""
        if self.budget_usd is None:
            return True
        elapsed = (time.monotonic() - self.started) / self.session_seconds if self.session_seconds else 1
        allowed = self.budget_usd * min(1.0, elapsed + 0.1)
        return self.cost_usd < allowed

    @property
    def cost_usd(self) -> float:
        pin, pout = PRICES.get(self.model, (4.0, 20.0))
        return (self.input_tokens * pin + self.output_tokens * pout) / 1_000_000

    def ask(self, role_prompt: str, data: str, schema: type[T]) -> T | None:
        """Возвращает объект schema или None (тогда агент использует запасной алгоритм)."""
        if not self.can_spend():
            self.skipped += 1
            return None
        try:
            response = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=16000,
                # если модель откажет по соображениям безопасности - API сам повторит на запасной
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=[{
                    "type": "text",
                    "text": f"{COMMON_RULES}\n\n{role_prompt}",
                    "cache_control": {"type": "ephemeral"},
                }],
                output_config={"effort": self.effort},
                output_format=schema,
                messages=[{"role": "user", "content": data}],
            )
        except anthropic.AuthenticationError:
            log.error("Claude: неверный ANTHROPIC_API_KEY")
            self.errors += 1
            return None
        except anthropic.RateLimitError:
            log.warning("Claude: превышен лимит запросов, агент работает по алгоритму")
            self.errors += 1
            return None
        except anthropic.APIStatusError as e:
            log.warning("Claude: ошибка API %s: %s", e.status_code, e.message)
            self.errors += 1
            return None
        except anthropic.APIConnectionError:
            log.warning("Claude: нет связи с API")
            self.errors += 1
            return None

        usage = response.usage
        with self._lock:
            self.calls += 1
            # кэш считаем по полной цене входа - оценка расходов получается с запасом
            self.input_tokens += (usage.input_tokens or 0) + (usage.cache_read_input_tokens or 0) \
                + (usage.cache_creation_input_tokens or 0)
            self.output_tokens += usage.output_tokens or 0

        if response.stop_reason == "refusal":
            log.warning("Claude отказался отвечать (request_id=%s)", response._request_id)
            return None
        if response.stop_reason == "max_tokens":
            log.warning("Claude: ответ обрезан по max_tokens")
            return None
        return response.parsed_output
