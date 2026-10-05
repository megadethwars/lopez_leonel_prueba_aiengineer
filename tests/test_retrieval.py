"""RAG: la recuperación trae el documento correcto y no trae nada si no hay relevancia."""

import pytest


@pytest.mark.parametrize(
    "question, expected_doc",
    [
        ("¿Cuánto dura la garantía de mi refrigeradora?", "doc1"),
        ("¿Puedo devolver una plancha que compré hace 2 semanas?", "doc2"),
        ("¿Cuánto tarda un envío a otra ciudad?", "doc3"),
        ("¿Cuándo me devuelven mi dinero?", "doc4"),
        ("Quiero poner una queja de un empleado que me trató mal", "doc5"),
        # Consultas cortas: el componente léxico del score híbrido las rescata.
        ("¿Garantía de una licuadora?", "doc1"),
        ("devoluciones", "doc2"),
        ("Mi tostadora no funciona", "doc1"),
    ],
)
def test_top_document_is_the_right_policy(retriever, question, expected_doc):
    results = retriever.retrieve(question)
    assert results, "debería recuperar al menos un documento sobre el umbral"
    assert results[0].document.doc_id == expected_doc


def test_warranty_question_retrieves_warranty_policy_text(retriever):
    top = retriever.retrieve("¿Qué garantía tiene una licuadora?")[0]
    assert top.document.title == "Política de garantía"
    assert "6 meses" in top.document.content


def test_results_respect_threshold_and_top_k(retriever):
    results = retriever.retrieve("¿Cuánto dura la garantía de mi refrigeradora?")
    assert len(results) <= retriever.top_k
    assert results[0].score >= retriever.threshold  # puerta: la pregunta es del dominio
    assert all(r.score >= retriever.context_threshold for r in results)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


@pytest.mark.parametrize(
    "question",
    ["Recomiéndame una receta de pastel", "¿Venden televisores?", "¿Quién ganó el mundial?", "hola"],
)
def test_out_of_domain_question_retrieves_nothing(retriever, question):
    assert retriever.retrieve(question) == []


def test_related_policies_enter_the_context(retriever):
    # Una pregunta de reembolso también necesita la Política de devoluciones (umbral de contexto).
    ids = [r.document.doc_id for r in retriever.retrieve("Quiero un reembolso de $300 por mi plancha")]
    assert ids[0] == "doc4" and "doc2" in ids
    # Devolver fuera de los 30 días depende de la garantía.
    ids = [r.document.doc_id for r in retriever.retrieve("¿Puedo devolver algo que compré hace 40 días?")]
    assert ids[0] == "doc2" and "doc1" in ids
