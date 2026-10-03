"""La base de conocimiento se carga completa y sin alterar desde data/knowledge_base/."""

import pytest

from tiendahogar_agent.adapters.outbound.knowledge.filesystem_knowledge_source import (
    FileSystemKnowledgeSource,
)
from tiendahogar_agent.config import settings


def _load(directory=None):
    return FileSystemKnowledgeSource(directory or settings.knowledge_base_dir).list_documents()


def test_loads_the_five_documents_in_order():
    docs = _load()
    assert [d.doc_id for d in docs] == ["doc1", "doc2", "doc3", "doc4", "doc5"]
    assert [d.title for d in docs] == [
        "Política de garantía",
        "Política de devoluciones",
        "Tiempos de envío",
        "Reembolsos",
        "Canales de contacto",
    ]


def test_document_text_is_verbatim():
    refunds = {d.doc_id: d for d in _load()}["doc4"]
    assert refunds.content == (
        "Los reembolsos se procesan en 5-10 días hábiles después de recibir el producto "
        "devuelto. Se reembolsa al mismo método de pago original. Reembolsos mayores a $500 "
        "requieren aprobación de un supervisor humano — el agente no debe aprobarlos "
        "automáticamente."
    )


def test_malformed_document_is_rejected(tmp_path):
    (tmp_path / "doc9_sin_texto.txt").write_text("Solo un título, sin texto", encoding="utf-8")
    with pytest.raises(ValueError):
        _load(tmp_path)
