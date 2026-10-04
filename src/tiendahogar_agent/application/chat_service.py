"""Caso de uso de chat: un turno del agente + persistencia del historial.

Envuelve a `SupportAgent` sin modificar su lógica: guarda cada turno vía
`ConversationRepositoryPort` y, si el agente perdió la memoria de una sesión (p. ej.
tras reiniciar el servicio), le devuelve el contexto desde el historial.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from ..domain.conversation import Conversation, ConversationSummary
from ..ports.conversations import ConversationRepositoryPort
from .support_agent import SupportAgent

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ChatService:
    def __init__(self, agent: SupportAgent, conversations: ConversationRepositoryPort):
        self._agent = agent
        self._conversations = conversations

    def ask(self, question: str, session_id: str) -> dict[str, Any]:
        conversation = self._conversations.get(session_id)
        prior = conversation.turns() if conversation else None
        result = self._agent.ask(question, session_id, prior_turns=prior)
        self._record(conversation, session_id, question, result)
        return result

    def _record(self, conversation: Conversation | None, session_id: str, question: str,
                result: dict) -> None:
        # Un fallo al guardar el historial no debe dejar al cliente sin respuesta.
        try:
            now = _now()
            conversation = conversation or Conversation.start(session_id, question, now)
            conversation.add_turn(question, result, now)
            self._conversations.save(conversation)
        except Exception:
            logger.exception("No se pudo guardar el historial de la conversación %s", session_id)

    def list_conversations(self) -> list[ConversationSummary]:
        return self._conversations.list_summaries()

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        return self._conversations.get(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        deleted = self._conversations.delete(conversation_id)
        if deleted:
            logger.info("Conversación %s eliminada", conversation_id)
        return deleted
