"""Orquestación end-to-end del grafo con un LLM falso (sin red ni API key)."""

import json
import uuid

from conftest import FakeLLM, RecordingNotifier, make_agent, text_response, tool_use_response
from tiendahogar_agent.application.prompts import NO_CONTEXT_MESSAGE

EMAIL = "soporte@tiendahogar.example"


def _ask(agent, question, session_id=None):
    return agent.ask(question, session_id=session_id or str(uuid.uuid4()))


def test_guardrail_escalates_refund_over_500_without_calling_llm(retriever):
    llm = FakeLLM([])
    result = _ask(make_agent(llm, retriever), "Quiero un reembolso de $900 por mi refrigeradora")
    assert result["escalated"] is True
    assert result["escalation"]["category"] == "reembolso_mayor_500"
    assert EMAIL in result["answer"]
    assert llm.calls == []  # el LLM nunca vio el caso


def test_order_question_uses_tool_and_feeds_real_data_to_llm(retriever):
    llm = FakeLLM([
        tool_use_response("consultar_estado_pedido", {"order_id": "ORD-1001"}),
        text_response("Tu Refrigeradora está En tránsito; llega en 3 días hábiles."),
    ])
    result = _ask(make_agent(llm, retriever), "¿Dónde está mi pedido ORD-1001?")

    assert result["tool_calls"][0]["output"]["estado"] == "En tránsito"
    # La segunda llamada al LLM recibe el tool_result con los datos de la tabla.
    tool_result = llm.calls[1][-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert json.loads(tool_result["content"])["producto"] == "Refrigeradora"
    assert result["escalated"] is False


def test_unknown_order_reaches_llm_as_not_found(retriever):
    llm = FakeLLM([
        tool_use_response("consultar_estado_pedido", {"order_id": "ORD-9999"}),
        text_response("No encontré ningún pedido con el ID ORD-9999."),
    ])
    result = _ask(make_agent(llm, retriever), "¿Y el pedido ORD-9999?")
    output = result["tool_calls"][0]["output"]
    assert output["encontrado"] is False and "producto" not in output


def test_out_of_domain_question_answers_no_info_without_llm(retriever):
    llm = FakeLLM([])
    result = _ask(make_agent(llm, retriever), "Recomiéndame una receta de pastel")
    assert result["answer"] == NO_CONTEXT_MESSAGE
    assert result["sources"] == [] and llm.calls == []


def test_rag_context_is_sent_to_llm(retriever):
    llm = FakeLLM([text_response("Las licuadoras tienen garantía de 6 meses.")])
    result = _ask(make_agent(llm, retriever), "¿Qué garantía tiene una licuadora?")
    assert result["sources"][0]["doc_id"] == "doc1"
    prompt = llm.calls[0][-1]["content"]
    assert '<documento id="doc1"' in prompt and "6 meses" in prompt


def test_llm_escalation_tool_forces_human_channel(retriever):
    # Caso que las reglas no capturan (paráfrasis): el LLM escala vía tool.
    llm = FakeLLM([
        tool_use_response("escalar_a_humano", {"categoria": "disputa_facturacion", "motivo": "monto"}),
        text_response("Entiendo tu situación."),  # olvida el canal → output guardrail lo añade
    ])
    result = _ask(make_agent(llm, retriever), "El total que pagué no coincide con lo que compré")
    assert result["escalated"] is True
    assert result["escalation"]["source"] == "llm_tool"
    assert EMAIL in result["answer"]


def test_output_guardrail_blocks_llm_refund_approval(retriever):
    llm = FakeLLM([text_response("¡Listo! Tu reembolso ha sido aprobado.")])
    result = _ask(make_agent(llm, retriever), "¿Cuándo me devuelven mi dinero?")
    assert result["escalated"] is True
    assert "aprobado" not in result["answer"]


def test_conversation_history_is_append_only_across_turns(retriever):
    llm = FakeLLM([text_response("Garantía de 12 meses."), text_response("Sí, dentro de 30 días.")])
    agent, sid = make_agent(llm, retriever), "s1"
    _ask(agent, "¿Garantía de una lavadora?", sid)
    first_turn = llm.calls[0]
    _ask(agent, "¿Y puedo devolverla?", sid)
    # El segundo request contiene el primero intacto como prefijo (append-only).
    assert llm.calls[1][: len(first_turn)] == first_turn
    assert llm.calls[1][len(first_turn)]["role"] == "assistant"


def test_escalation_is_published_through_notifier_port(retriever):
    notifier = RecordingNotifier()
    agent = make_agent(FakeLLM([]), retriever, notifier=notifier)
    _ask(agent, "Voy a demandar a la tienda", session_id="s-esc")
    assert len(notifier.events) == 1
    event = notifier.events[0]
    assert event["session_id"] == "s-esc"
    assert event["escalation"].category.value == "tema_legal"


def test_notifier_failure_does_not_break_the_answer(retriever):
    class BrokenNotifier:
        def notify(self, escalation, session_id, question):
            raise ConnectionError("broker caído")

    result = _ask(make_agent(FakeLLM([]), retriever, notifier=BrokenNotifier()),
                  "Voy a demandar a la tienda")
    assert result["escalated"] is True and EMAIL in result["answer"]


def test_no_notification_when_not_escalated(retriever):
    notifier = RecordingNotifier()
    llm = FakeLLM([text_response("Las licuadoras tienen 6 meses de garantía.")])
    _ask(make_agent(llm, retriever, notifier=notifier), "¿Qué garantía tiene una licuadora?")
    assert notifier.events == []
