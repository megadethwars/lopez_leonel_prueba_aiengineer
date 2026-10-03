"""Configuración de logging del agente.

- Nivel con LOG_LEVEL (default INFO). DEBUG muestra además los scores de todos los documentos.
- LOG_FORMAT=json emite una línea JSON por evento (para Azure Monitor / Datadog / ELK).
- Cada registro lleva el `session_id` de la conversación en curso (contextvar), de modo
  que se puede seguir un turno completo: request → guardrail → RAG → LLM → tools → respuesta.
- Los textos del cliente se truncan (`preview`): en producción deberían enmascararse PII.
"""

import contextvars
import json
import logging
import os

session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")

_PACKAGE = "tiendahogar_agent"
_TEXT_FORMAT = "%(asctime)s %(levelname)-7s [%(session_id)s] %(name)s: %(message)s"


class _SessionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = session_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "session_id": getattr(record, "session_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str | None = None) -> None:
    """Configura el logger del paquete (idempotente)."""
    logger = logging.getLogger(_PACKAGE)
    logger.setLevel((level or os.getenv("LOG_LEVEL") or "INFO").upper())
    if any(getattr(h, "_tiendahogar", False) for h in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler._tiendahogar = True  # type: ignore[attr-defined]
    handler.addFilter(_SessionFilter())
    if (os.getenv("LOG_FORMAT") or "text").lower() == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))
    logger.addHandler(handler)


def preview(text: str, limit: int = 80) -> str:
    """Versión corta de un texto de usuario para logs."""
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
