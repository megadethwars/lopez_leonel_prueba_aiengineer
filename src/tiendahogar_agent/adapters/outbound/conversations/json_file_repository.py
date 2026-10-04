"""Adapter de `ConversationRepositoryPort`: un archivo JSON por conversación.

    conversations/<conversation_id>.json

- Escritura atómica (archivo temporal + reemplazo): un corte a mitad de escritura no
  deja un JSON corrupto.
- Solo acepta IDs con el formato del dominio, así que nunca se construye una ruta
  fuera de la carpeta (path traversal).
- En producción se reemplaza por un adapter de base de datos (Cosmos DB, PostgreSQL…).
"""

import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path

from ....domain.conversation import (
    Conversation,
    ConversationMessage,
    ConversationSummary,
    is_valid_conversation_id,
)

logger = logging.getLogger(__name__)


def _to_dict(conversation: Conversation) -> dict:
    return asdict(conversation)


def _from_dict(data: dict) -> Conversation:
    messages = [ConversationMessage(**m) for m in data.get("messages", [])]
    return Conversation(data["id"], data["title"], data["created_at"], data["updated_at"], messages)


class JsonFileConversationRepository:
    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, conversation_id: str) -> Path | None:
        return self.directory / f"{conversation_id}.json" if is_valid_conversation_id(conversation_id) else None

    def get(self, conversation_id: str) -> Conversation | None:
        path = self._path(conversation_id)
        if path is None or not path.is_file():
            return None
        try:
            return _from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, KeyError, TypeError):
            logger.warning("Archivo de conversación ilegible: %s", path.name)
            return None

    def save(self, conversation: Conversation) -> None:
        path = self._path(conversation.id)
        if path is None:
            raise ValueError(f"ID de conversación inválido: {conversation.id!r}")
        payload = json.dumps(_to_dict(conversation), ensure_ascii=False, indent=2)
        with self._lock:
            fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(payload)
                os.replace(tmp, path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

    def list_summaries(self) -> list[ConversationSummary]:
        summaries = []
        for path in self.directory.glob("*.json"):
            conversation = self.get(path.stem)
            if conversation is not None:
                summaries.append(conversation.summary())
        return sorted(summaries, key=lambda s: s.updated_at, reverse=True)

    def delete(self, conversation_id: str) -> bool:
        path = self._path(conversation_id)
        if path is None or not path.is_file():
            return False
        with self._lock:
            path.unlink(missing_ok=True)
        return True
