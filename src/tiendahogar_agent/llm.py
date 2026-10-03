"""Cliente LLM (Claude vía SDK oficial de Anthropic) detrás de una interfaz mínima.

La interfaz `LLMClient` permite inyectar un LLM falso en los tests, de modo que
la orquestación (grafo, tools, guardrails) se prueba sin red ni API key.
"""

import logging
import time
from typing import Any, Protocol

import anthropic

from .config import settings

logger = logging.getLogger(__name__)

LLMResponse = dict[str, Any]  # {"stop_reason": str, "content": list[dict], "model": str}


class LLMNotConfiguredError(RuntimeError):
    """No hay credenciales para el proveedor LLM."""


class LLMClient(Protocol):
    def create(
        self, system: str, messages: list[dict], tools: list[dict]
    ) -> LLMResponse: ...


def _drop_declined_partial(content: list[dict]) -> list[dict]:
    """Si hubo fallback server-side a mitad de la respuesta, omite los bloques
    thinking/tool_use previos al último bloque `fallback` (regla de eco de la API)."""
    last = max((i for i, b in enumerate(content) if b.get("type") == "fallback"), default=None)
    if last is None:
        return content
    dropped = {"thinking", "redacted_thinking", "tool_use"}
    head = [b for b in content[:last] if b.get("type") not in dropped]
    return head + content[last + 1 :]


class AnthropicLLM:
    def __init__(
        self,
        model: str | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        client: anthropic.Anthropic | None = None,
    ):
        self.model = model or settings.anthropic_model
        self.effort = effort or settings.anthropic_effort
        self.max_tokens = max_tokens or settings.max_tokens
        self.client = client or anthropic.Anthropic()

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        if not (self.client.api_key or self.client.auth_token):
            logger.error("No hay ANTHROPIC_API_KEY configurada; no se puede llamar al LLM")
            raise LLMNotConfiguredError("Define ANTHROPIC_API_KEY (ver .env.example).")
        # Thinking adaptativo (default del modelo) con esfuerzo bajo: es un chat de soporte
        # con respuestas cortas. `fallbacks="default"` reintenta en otro modelo si el
        # clasificador de seguridad rechaza la petición (solo Claude API).
        started = time.perf_counter()
        logger.info("LLM request: model=%s effort=%s mensajes=%d", self.model, self.effort, len(messages))
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=messages,
                tools=tools,
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError:
            logger.error("LLM rate limit (429) tras %.0f ms", (time.perf_counter() - started) * 1000)
            raise
        except anthropic.APIStatusError as e:
            logger.error("LLM error HTTP %s: %s", e.status_code, e.message)
            raise
        except anthropic.APIConnectionError as e:
            logger.error("LLM sin conexión: %s", e)
            raise
        content = [b.model_dump(mode="json", exclude_none=True) for b in response.content]
        usage = response.usage
        logger.info(
            "LLM response: model=%s stop_reason=%s tokens_in=%s tokens_out=%s cache_read=%s (%.0f ms)",
            response.model, response.stop_reason, usage.input_tokens, usage.output_tokens,
            usage.cache_read_input_tokens or 0, (time.perf_counter() - started) * 1000,
        )
        if any(b.get("type") == "fallback" for b in content):
            logger.warning("El modelo %s declinó; respondió el fallback %s", self.model, response.model)
        return {
            "stop_reason": response.stop_reason,
            "content": _drop_declined_partial(content),
            "model": response.model,
        }
