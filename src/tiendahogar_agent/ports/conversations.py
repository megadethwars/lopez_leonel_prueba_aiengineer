"""Port del historial de conversaciones.

Adapters posibles: archivos JSON (actual), Cosmos DB, PostgreSQL, Redis, etc.
"""

from typing import Protocol

from ..domain.conversation import Conversation, ConversationSummary


class ConversationRepositoryPort(Protocol):
    def get(self, conversation_id: str) -> Conversation | None:
        """La conversación con ese ID, o None si no existe (o el ID es inválido)."""
        ...

    def save(self, conversation: Conversation) -> None:
        """Crea o reemplaza la conversación."""
        ...

    def list_summaries(self) -> list[ConversationSummary]:
        """Resumen de todas las conversaciones, de la más reciente a la más antigua."""
        ...

    def delete(self, conversation_id: str) -> bool:
        """Elimina la conversación; True si existía."""
        ...
