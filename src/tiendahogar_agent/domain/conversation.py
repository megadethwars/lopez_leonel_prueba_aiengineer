"""Historial de conversaciones: entidades y reglas de negocio (sin I/O)."""

import re
from dataclasses import dataclass, field

TITLE_MAX_CHARS = 60
# IDs seguros para cualquier almacenamiento (archivos, claves, URLs): evita path traversal.
CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def is_valid_conversation_id(conversation_id: str) -> bool:
    return bool(CONVERSATION_ID_PATTERN.fullmatch(conversation_id or ""))


def title_from(question: str) -> str:
    """Título de la conversación: la primera pregunta del cliente, recortada."""
    text = " ".join((question or "").split())
    if len(text) <= TITLE_MAX_CHARS:
        return text or "Conversación"
    return text[: TITLE_MAX_CHARS - 1].rstrip() + "…"


@dataclass
class ConversationMessage:
    role: str  # "user" | "assistant"
    content: str
    timestamp: str  # ISO 8601 (UTC)
    sources: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    escalation: dict | None = None


@dataclass(frozen=True)
class ConversationSummary:
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


@dataclass
class Conversation:
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[ConversationMessage] = field(default_factory=list)

    @classmethod
    def start(cls, conversation_id: str, first_question: str, now: str) -> "Conversation":
        if not is_valid_conversation_id(conversation_id):
            raise ValueError(f"ID de conversación inválido: {conversation_id!r}")
        return cls(conversation_id, title_from(first_question), now, now)

    def add_turn(self, question: str, result: dict, now: str) -> None:
        """Agrega la pregunta del cliente y la respuesta del agente (con su metadata)."""
        self.messages.append(ConversationMessage("user", question, now))
        self.messages.append(ConversationMessage(
            "assistant", result["answer"], now,
            sources=list(result.get("sources") or []),
            tool_calls=list(result.get("tool_calls") or []),
            escalation=result.get("escalation"),
        ))
        self.updated_at = now

    def turns(self) -> list[tuple[str, str]]:
        """Pares (pregunta, respuesta) en orden, para devolver el contexto al agente."""
        pairs, pending = [], None
        for m in self.messages:
            if m.role == "user":
                pending = m.content
            elif m.role == "assistant" and pending is not None:
                pairs.append((pending, m.content))
                pending = None
        return pairs

    def summary(self) -> ConversationSummary:
        return ConversationSummary(self.id, self.title, self.created_at, self.updated_at,
                                   len(self.messages))
