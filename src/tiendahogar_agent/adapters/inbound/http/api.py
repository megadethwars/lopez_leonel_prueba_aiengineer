"""Adapter de entrada HTTP (FastAPI).

Solo traduce HTTP ⇄ casos de uso: no conoce el LLM, el SDK del proveedor ni dónde se
guardan las conversaciones; los errores del LLM llegan como `LLMError` (port) y se
mapean a códigos HTTP.
"""

import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from ....application.chat_service import ChatService
from ....domain.conversation import CONVERSATION_ID_PATTERN
from ....logging_config import session_id_var
from ....ports.llm import LLMConnectionError, LLMNotConfiguredError, LLMProviderError, LLMRateLimitError

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
_ID_REGEX = CONVERSATION_ID_PATTERN.pattern


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000,
                         examples=["¿Cuánto dura la garantía de una licuadora?"])
    session_id: str | None = Field(None, pattern=_ID_REGEX,
                                   description="ID de la conversación. Omitir para iniciar una nueva.")


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


class ConversationSummaryOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class MessageOut(BaseModel):
    role: str
    content: str
    timestamp: str
    sources: list[dict]
    tool_calls: list[dict]
    escalation: dict | None


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[MessageOut]


def create_app(service_factory: Callable[[], ChatService]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Construye el agente (embeddings, índice, skill) una sola vez al arrancar.
        logger.info("Iniciando microservicio...")
        app.state.chat = service_factory()
        logger.info("Microservicio listo (auth X-API-Key %s)",
                    "activada" if os.getenv("DEMO_API_KEY") else "desactivada")
        yield
        logger.info("Microservicio detenido")

    app = FastAPI(
        title="TiendaHogar — Agente de soporte",
        version="1.1.0",
        description="Agente RAG con tool de pedidos, guardrails de escalamiento a humano e historial de conversaciones.",
        lifespan=lifespan,
    )
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def require_api_key(key: str | None = Security(api_key_header)) -> None:
        # Si DEMO_API_KEY está definida (despliegue público), se exige el header X-API-Key.
        expected = os.getenv("DEMO_API_KEY")
        if expected and not (key and secrets.compare_digest(key, expected)):
            logger.warning("Petición rechazada: X-API-Key inválida o ausente")
            raise HTTPException(status_code=401, detail="API key inválida o ausente.")

    def chat_service(request: Request) -> ChatService:
        return request.app.state.chat

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
    def chat(body: ChatRequest, service: ChatService = Depends(chat_service)) -> dict:
        session_id = body.session_id or uuid.uuid4().hex
        token = session_id_var.set(session_id)
        try:
            logger.info("POST /chat (%s sesión)", "nueva" if body.session_id is None else "continúa")
            return service.ask(body.message, session_id=session_id)
        except LLMNotConfiguredError as e:
            logger.error("503: LLM no configurado")
            raise HTTPException(status_code=503, detail=f"LLM no configurado: {e}")
        except LLMRateLimitError:
            logger.warning("429: rate limit del proveedor LLM")
            raise HTTPException(status_code=429, detail="El servicio está saturado; intenta en unos segundos.")
        except LLMProviderError as e:
            logger.error("502: error HTTP %s del proveedor LLM", e.status_code)
            raise HTTPException(status_code=502, detail=f"Error del proveedor LLM ({e.status_code}).")
        except LLMConnectionError:
            logger.error("503: sin conexión con el proveedor LLM")
            raise HTTPException(status_code=503, detail="No se pudo contactar al proveedor LLM.")
        finally:
            session_id_var.reset(token)

    @app.get("/conversations", response_model=list[ConversationSummaryOut],
             dependencies=[Depends(require_api_key)])
    def list_conversations(service: ChatService = Depends(chat_service)) -> list:
        return [s.__dict__ for s in service.list_conversations()]

    @app.get("/conversations/{conversation_id}", response_model=ConversationOut,
             dependencies=[Depends(require_api_key)])
    def get_conversation(conversation_id: str, service: ChatService = Depends(chat_service)):
        conversation = service.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversación no encontrada.")
        return conversation

    @app.delete("/conversations/{conversation_id}", status_code=204,
                dependencies=[Depends(require_api_key)])
    def delete_conversation(conversation_id: str, service: ChatService = Depends(chat_service)):
        if not service.delete_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversación no encontrada.")
        return Response(status_code=204)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")

    return app
