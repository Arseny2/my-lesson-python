"""Обёртка над Claude API: один вызов = один ответ строго по схеме (pydantic)."""
from __future__ import annotations

import logging
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
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium"):
        # Ключ берётся из ANTHROPIC_API_KEY (или профиля `ant auth login`)
        self.client = anthropic.Anthropic(max_retries=3)
        self.model = model
        self.effort = effort
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.errors = 0

    @property
    def cost_usd(self) -> float:
        pin, pout = PRICES.get(self.model, (4.0, 20.0))
        return (self.input_tokens * pin + self.output_tokens * pout) / 1_000_000

    def ask(self, role_prompt: str, data: str, schema: type[T]) -> T | None:
        """Возвращает объект schema или None (тогда агент использует запасной алгоритм)."""
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

        self.calls += 1
        usage = response.usage
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
