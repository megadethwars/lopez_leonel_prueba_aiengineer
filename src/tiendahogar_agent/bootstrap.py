"""Composition root: el único lugar que conoce a la vez los ports y sus adapters.

Para conectar un sistema nuevo (otra base de datos, un microservicio, Kafka, otro
proveedor LLM) se escribe un adapter que cumpla el port correspondiente y se
cambia aquí la línea que lo construye. Dominio y aplicación no se tocan.

    Port                     Adapter actual                      Ejemplo en producción
    LLMPort                  AnthropicLLM                        Claude en Microsoft Foundry
    KnowledgeSourcePort      FileSystemKnowledgeSource (.txt)    Tablas Delta en Unity Catalog
    RetrieverPort            HybridRetriever (en memoria)        Mosaic AI Vector Search
    OrderRepositoryPort      InMemoryOrderRepository (mock)      API del OMS detrás de Apigee
    SkillRepositoryPort      FileSystemSkillRepository (.md)     Servicio de gestión de prompts
    EscalationNotifierPort   LoggingEscalationNotifier           Productor Kafka
    ConversationRepositoryPort  JsonFileConversationRepository   Cosmos DB / PostgreSQL
"""

from functools import lru_cache
from typing import Any

from .adapters.outbound.conversations.json_file_repository import JsonFileConversationRepository
from .adapters.outbound.knowledge.filesystem_knowledge_source import FileSystemKnowledgeSource
from .adapters.outbound.llm.anthropic_llm import AnthropicLLM
from .adapters.outbound.notifications.logging_escalation_notifier import LoggingEscalationNotifier
from .adapters.outbound.orders.in_memory_order_repository import InMemoryOrderRepository
from .adapters.outbound.retrieval.embeddings import FastEmbedEmbedder
from .adapters.outbound.retrieval.hybrid_retriever import HybridRetriever
from .adapters.outbound.skills.filesystem_skill_repository import FileSystemSkillRepository
from .application.chat_service import ChatService
from .application.order_status import OrderStatusService
from .application.prompts import build_system_prompt
from .application.support_agent import SupportAgent
from .application.tools import build_tool_registry
from .config import Settings, settings
from .ports import (
    ConversationRepositoryPort,
    EscalationNotifierPort,
    KnowledgeSourcePort,
    LLMPort,
    OrderRepositoryPort,
    RetrieverPort,
    SkillRepositoryPort,
)


@lru_cache(maxsize=1)
def build_retriever(config: Settings = settings) -> RetrieverPort:
    """Índice RAG (costoso: carga el modelo de embeddings). Se construye una vez por proceso."""
    return build_retriever_from(FileSystemKnowledgeSource(config.knowledge_base_dir), config)


def build_retriever_from(source: KnowledgeSourcePort, config: Settings = settings) -> RetrieverPort:
    return HybridRetriever(
        embedder=FastEmbedEmbedder(config.embedding_model),
        documents=source.list_documents(),
        top_k=config.retrieval_top_k,
        threshold=config.retrieval_threshold,
        alpha=config.retrieval_alpha,
    )


def build_support_agent(
    *,
    llm: LLMPort | None = None,
    retriever: RetrieverPort | None = None,
    order_repository: OrderRepositoryPort | None = None,
    skill_repository: SkillRepositoryPort | None = None,
    notifier: EscalationNotifierPort | None = None,
    checkpointer: Any | None = None,
    config: Settings = settings,
) -> SupportAgent:
    """Arma el agente con los adapters por defecto; cualquier port se puede sustituir."""
    order_service = OrderStatusService(order_repository or InMemoryOrderRepository())
    skills = skill_repository or FileSystemSkillRepository(config.skills_dir)
    return SupportAgent(
        llm=llm or AnthropicLLM(config.anthropic_model, config.anthropic_effort, config.max_tokens),
        retriever=retriever or build_retriever(config),
        tools=build_tool_registry(order_service),
        system_prompt=build_system_prompt(skills.get(config.conversation_skill)),
        notifier=notifier or LoggingEscalationNotifier(),
        checkpointer=checkpointer,
        max_tool_iterations=config.max_tool_iterations,
    )


def build_chat_service(
    *,
    agent: SupportAgent | None = None,
    conversation_repository: ConversationRepositoryPort | None = None,
    config: Settings = settings,
) -> ChatService:
    """Caso de uso de chat (agente + historial de conversaciones)."""
    return ChatService(
        agent=agent or build_support_agent(config=config),
        conversations=conversation_repository or JsonFileConversationRepository(config.conversations_dir),
    )


# Tool requerida por la especificación, con la firma exacta
# `consultar_estado_pedido(order_id: str) -> dict`, sobre el repositorio por defecto.
consultar_estado_pedido = OrderStatusService(InMemoryOrderRepository()).consultar_estado_pedido
