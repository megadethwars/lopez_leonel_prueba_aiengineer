"""Los eventos clave del agente quedan registrados con el session_id del turno."""

import logging

from conftest import FakeLLM, make_agent, text_response, tool_use_response


def _messages(caplog, session_id):
    return [r.getMessage() for r in caplog.records
            if r.name.startswith("tiendahogar_agent") and getattr(r, "session_id", None) == session_id]


def _attach_session_filter(caplog):
    from tiendahogar_agent.logging_config import _SessionFilter

    caplog.handler.addFilter(_SessionFilter())


def test_tool_call_and_rag_are_logged(retriever, caplog):
    _attach_session_filter(caplog)
    caplog.set_level(logging.INFO, logger="tiendahogar_agent")
    llm = FakeLLM([
        tool_use_response("consultar_estado_pedido", {"order_id": "ORD-1001"}),
        text_response("Tu pedido está en tránsito."),
    ])
    make_agent(llm, retriever).ask("¿Dónde está mi pedido ORD-1001?", session_id="log-1")

    logs = _messages(caplog, "log-1")
    assert any(m.startswith("Pregunta recibida") for m in logs)
    assert any(m.startswith("RAG:") for m in logs)
    assert any(m.startswith("Tool consultar_estado_pedido") and "En tránsito" in m for m in logs)
    assert any(m.startswith("Turno completado") for m in logs)


def test_guardrail_escalation_is_logged_as_warning(retriever, caplog):
    _attach_session_filter(caplog)
    caplog.set_level(logging.INFO, logger="tiendahogar_agent")
    make_agent(FakeLLM([]), retriever).ask("Voy a demandar a la tienda", session_id="log-2")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING
                and getattr(r, "session_id", None) == "log-2"]
    assert any("Guardrail de entrada activado" in r.getMessage() and "tema_legal" in r.getMessage()
               for r in warnings)
