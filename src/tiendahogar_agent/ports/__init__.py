"""Ports: contratos que el núcleo (application) necesita del mundo exterior.

Cada port es un `typing.Protocol`; cualquier clase con los mismos métodos lo
implementa (tipado estructural), sin heredar. Los adapters viven en `adapters/`
y se conectan en `bootstrap.py`. Para integrar un sistema nuevo (otra base de
datos, una API, un broker) se escribe un adapter que cumpla el port, sin tocar
el núcleo.
"""

from .conversations import ConversationRepositoryPort
from .knowledge import KnowledgeSourcePort
from .llm import (
    LLMConnectionError,
    LLMError,
    LLMNotConfiguredError,
    LLMPort,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
)
from .notifications import EscalationNotifierPort
from .orders import OrderRepositoryPort
from .retrieval import RetrieverPort
from .skills import SkillRepositoryPort

__all__ = [
    "ConversationRepositoryPort",
    "EscalationNotifierPort",
    "KnowledgeSourcePort",
    "LLMConnectionError",
    "LLMError",
    "LLMNotConfiguredError",
    "LLMPort",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMResponse",
    "OrderRepositoryPort",
    "RetrieverPort",
    "SkillRepositoryPort",
]
