"""Microservicio FastAPI que expone el agente.

    uvicorn tiendahogar_agent.api:app --app-dir src --port 8000
"""

import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .logging_config import session_id_var, setup_logging

setup_logging()  # antes de importar el grafo: así se registra también la carga de documentos

from .graph import SupportAgent  # noqa: E402
from .llm import LLMNotConfiguredError  # noqa: E402

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, examples=["¿Cuánto dura la garantía de una licuadora?"])
    session_id: str | None = Field(None, description="Omitir para iniciar una conversación nueva.")


class Source(BaseModel):
    doc_id: str
    title: str
    score: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    escalated: bool
    escalation: dict | None
    sources: list[Source]
    tool_calls: list[dict]


def _build_default_agent() -> SupportAgent:
    from .llm import AnthropicLLM
    from .retriever import get_retriever

    return SupportAgent(llm=AnthropicLLM(), retriever=get_retriever())


def create_app(agent: SupportAgent | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Carga embeddings e índice una sola vez al arrancar (no en cada request).
        logger.info("Iniciando microservicio...")
        app.state.agent = agent or _build_default_agent()
        logger.info("Microservicio listo (auth X-API-Key %s)",
                    "activada" if os.getenv("DEMO_API_KEY") else "desactivada")
        yield
        logger.info("Microservicio detenido")

    app = FastAPI(
        title="TiendaHogar — Agente de soporte",
        version="1.0.0",
        description="Agente RAG con tool de pedidos y guardrails de escalamiento a humano.",
        lifespan=lifespan,
    )
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def require_api_key(key: str | None = Security(api_key_header)) -> None:
        # Si DEMO_API_KEY está definida (despliegue público), se exige el header X-API-Key.
        expected = os.getenv("DEMO_API_KEY")
        if expected and not (key and secrets.compare_digest(key, expected)):
            logger.warning("Petición rechazada: X-API-Key inválida o ausente")
            raise HTTPException(status_code=401, detail="API key inválida o ausente.")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
    def chat(body: ChatRequest, request: Request) -> dict:
        session_id = body.session_id or str(uuid.uuid4())
        token = session_id_var.set(session_id)
        try:
            logger.info("POST /chat (%s sesión)", "nueva" if body.session_id is None else "continúa")
            return request.app.state.agent.ask(body.message, session_id=session_id)
        except LLMNotConfiguredError as e:
            logger.error("503: LLM no configurado")
            raise HTTPException(status_code=503, detail=f"LLM no configurado: {e}")
        except anthropic.RateLimitError:
            logger.warning("429: rate limit del proveedor LLM")
            raise HTTPException(status_code=429, detail="El servicio está saturado; intenta en unos segundos.")
        except anthropic.APIStatusError as e:
            logger.error("502: error HTTP %s del proveedor LLM", e.status_code)
            raise HTTPException(status_code=502, detail=f"Error del proveedor LLM ({e.status_code}).")
        except anthropic.APIConnectionError:
            logger.error("503: sin conexión con el proveedor LLM")
            raise HTTPException(status_code=503, detail="No se pudo contactar al proveedor LLM.")
        finally:
            session_id_var.reset(token)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

    return app


app = create_app()
