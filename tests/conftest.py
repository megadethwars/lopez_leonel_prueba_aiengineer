from typing import Any

import pytest

from tiendahogar_agent.bootstrap import build_retriever, build_support_agent


@pytest.fixture(scope="session")
def retriever():
    """Retriever real (embeddings multilingües locales; se descargan la primera vez)."""
    return build_retriever()


class FakeLLM:
    """Adapter falso de `LLMPort`: devuelve respuestas guionizadas y registra cada llamada.

    Permite probar la orquestación (tools, guardrails, rutas) sin red ni API key.
    """

    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []
        self.systems: list[str] = []

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        self.calls.append([dict(m) for m in messages])
        self.systems.append(system)
        return self.responses.pop(0)


class RecordingNotifier:
    """Adapter falso de `EscalationNotifierPort` que guarda los eventos."""

    def __init__(self):
        self.events: list[dict] = []

    def notify(self, escalation, session_id: str, question: str) -> None:
        self.events.append({"escalation": escalation, "session_id": session_id, "question": question})


def make_agent(llm, retriever, **overrides):
    """Agente real armado por el composition root, con el LLM (y otros ports) sustituidos."""
    return build_support_agent(llm=llm, retriever=retriever, **overrides)


def text_response(text: str) -> dict:
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}], "model": "fake"}


def tool_use_response(name: str, tool_input: dict, tool_id: str = "toolu_1") -> dict:
    return {
        "stop_reason": "tool_use",
        "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}],
        "model": "fake",
    }
