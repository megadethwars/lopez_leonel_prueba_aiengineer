"""Port del modelo de lenguaje.

El contrato usa el formato de mensajes de la API de Claude (bloques `text`,
`tool_use`, `tool_result`, `thinking`), que el núcleo trata como datos opacos.
Los errores del proveedor se traducen a la jerarquía `LLMError` para que los
adapters de entrada (HTTP, CLI) no dependan del SDK de ningún proveedor.
"""

from typing import Any, Protocol

LLMResponse = dict[str, Any]  # {"stop_reason": str, "content": list[dict], "model": str}


class LLMPort(Protocol):
    def create(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """Genera el siguiente turno del asistente."""
        ...


class LLMError(Exception):
    """Error genérico del proveedor LLM."""


class LLMNotConfiguredError(LLMError):
    """Faltan credenciales o configuración del proveedor."""


class LLMRateLimitError(LLMError):
    """El proveedor rechazó la petición por límite de uso (429)."""


class LLMProviderError(LLMError):
    """El proveedor respondió con un error HTTP."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class LLMConnectionError(LLMError):
    """No fue posible contactar al proveedor."""
