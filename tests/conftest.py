from typing import Any

import pytest

from tiendahogar_agent.retriever import get_retriever


@pytest.fixture(scope="session")
def retriever():
    """Retriever real (embeddings multilingües locales; se descargan la primera vez)."""
    return get_retriever()


class FakeLLM:
    """LLM guionizado: devuelve respuestas predefinidas y registra cada llamada.

    Permite probar la orquestación (tools, guardrails, rutas) sin red ni API key.
    """

    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        self.calls.append([dict(m) for m in messages])
        return self.responses.pop(0)


def text_response(text: str) -> dict:
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}], "model": "fake"}


def tool_use_response(name: str, tool_input: dict, tool_id: str = "toolu_1") -> dict:
    return {
        "stop_reason": "tool_use",
        "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}],
        "model": "fake",
    }
