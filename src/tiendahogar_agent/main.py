"""Punto de entrada ASGI del microservicio.

    uvicorn tiendahogar_agent.main:app --app-dir src --port 8000
"""

from .logging_config import setup_logging

setup_logging()  # antes de construir nada, para registrar también el arranque

from .adapters.inbound.http.api import create_app  # noqa: E402
from .bootstrap import build_chat_service  # noqa: E402

app = create_app(service_factory=build_chat_service)
