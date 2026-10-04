"""Historial de conversaciones: dominio, adapter JSON, caso de uso y endpoints HTTP."""

import json

import pytest
from fastapi.testclient import TestClient

from conftest import FakeLLM, make_agent, make_chat, text_response
from tiendahogar_agent.adapters.inbound.http.api import create_app
from tiendahogar_agent.adapters.outbound.conversations.json_file_repository import (
    JsonFileConversationRepository,
)
from tiendahogar_agent.application.chat_service import ChatService
from tiendahogar_agent.domain.conversation import Conversation, title_from

RESULT = {"answer": "Las licuadoras tienen 6 meses de garantía.", "escalated": False, "escalation": None,
          "sources": [{"doc_id": "doc1", "title": "Política de garantía", "score": 0.6}], "tool_calls": []}


# --- Dominio -----------------------------------------------------------------------------

def test_title_is_first_question_truncated():
    assert title_from("¿Garantía de una licuadora?") == "¿Garantía de una licuadora?"
    long = title_from("palabra " * 30)
    assert len(long) <= 60 and long.endswith("…")


def test_conversation_turns_pairs_questions_and_answers():
    conv = Conversation.start("c1", "hola", "2026-10-03T10:00:00+00:00")
    conv.add_turn("hola", {**RESULT, "answer": "¡Hola!"}, "2026-10-03T10:00:00+00:00")
    conv.add_turn("¿garantía?", RESULT, "2026-10-03T10:01:00+00:00")
    assert conv.turns() == [("hola", "¡Hola!"), ("¿garantía?", RESULT["answer"])]
    assert conv.updated_at == "2026-10-03T10:01:00+00:00"


def test_conversation_rejects_unsafe_ids():
    with pytest.raises(ValueError):
        Conversation.start("../../etc/passwd", "hola", "2026-10-03T10:00:00+00:00")


# --- Adapter JSON ---------------------------------------------------------------------------

def test_json_repository_round_trip(tmp_path):
    repo = JsonFileConversationRepository(tmp_path)
    conv = Conversation.start("abc-123", "¿Garantía?", "2026-10-03T10:00:00+00:00")
    conv.add_turn("¿Garantía?", RESULT, "2026-10-03T10:00:00+00:00")
    repo.save(conv)

    stored = json.loads((tmp_path / "abc-123.json").read_text(encoding="utf-8"))
    assert stored["title"] == "¿Garantía?" and len(stored["messages"]) == 2
    assert repo.get("abc-123") == conv
    assert [s.id for s in repo.list_summaries()] == ["abc-123"]
    assert repo.delete("abc-123") is True and repo.get("abc-123") is None
    assert repo.delete("abc-123") is False


def test_json_repository_lists_most_recent_first_and_skips_corrupt_files(tmp_path):
    repo = JsonFileConversationRepository(tmp_path)
    for cid, ts in [("old", "2026-10-01T10:00:00+00:00"), ("new", "2026-10-03T10:00:00+00:00")]:
        repo.save(Conversation.start(cid, "hola", ts))
    (tmp_path / "broken.json").write_text("{no es json", encoding="utf-8")
    assert [s.id for s in repo.list_summaries()] == ["new", "old"]


@pytest.mark.parametrize("bad_id", ["../secreto", "a/b", "", "x" * 65, "con espacios"])
def test_json_repository_never_touches_paths_outside_its_folder(tmp_path, bad_id):
    repo = JsonFileConversationRepository(tmp_path / "conversations")
    assert repo.get(bad_id) is None
    assert repo.delete(bad_id) is False


# --- Caso de uso -------------------------------------------------------------------------

def test_chat_service_records_each_turn_with_metadata(retriever, tmp_path):
    llm = FakeLLM([text_response("Las licuadoras tienen 6 meses de garantía.")])
    chat = make_chat(llm, retriever, tmp_path)
    chat.ask("¿Qué garantía tiene una licuadora?", session_id="s1")

    conv = chat.get_conversation("s1")
    assert conv.title == "¿Qué garantía tiene una licuadora?"
    assert [m.role for m in conv.messages] == ["user", "assistant"]
    assert conv.messages[1].sources[0]["doc_id"] == "doc1"


def test_chat_service_records_escalations(retriever, tmp_path):
    chat = make_chat(FakeLLM([]), retriever, tmp_path)
    chat.ask("Voy a demandar a la tienda", session_id="s-esc")
    assert chat.get_conversation("s-esc").messages[1].escalation["category"] == "tema_legal"


def test_conversation_continues_after_restart(retriever, tmp_path):
    # Primer "proceso": responde y guarda el historial.
    first = make_chat(FakeLLM([text_response("Las lavadoras tienen 12 meses de garantía.")]),
                      retriever, tmp_path)
    first.ask("¿Garantía de una lavadora?", session_id="s-restart")

    # Segundo "proceso": agente nuevo (sin memoria), mismo historial en disco.
    llm = FakeLLM([text_response("Sí, dentro de 30 días.")])
    second = make_chat(llm, retriever, tmp_path)
    second.ask("¿Y puedo devolverla?", session_id="s-restart")

    sent = llm.calls[0]
    assert "¿Garantía de una lavadora?" in sent[0]["content"]
    assert "sin documentos relevantes" not in sent[0]["content"]  # no induce falsas disculpas
    assert sent[1]["content"][0]["text"] == "Las lavadoras tienen 12 meses de garantía."
    assert len(second.get_conversation("s-restart").messages) == 4


def test_history_failure_does_not_break_the_answer(retriever):
    class BrokenRepository:
        def get(self, conversation_id): return None
        def save(self, conversation): raise OSError("disco lleno")
        def list_summaries(self): return []
        def delete(self, conversation_id): return False

    chat = ChatService(make_agent(FakeLLM([]), retriever), BrokenRepository())
    result = chat.ask("Voy a demandar a la tienda", session_id="s-broken")
    assert result["escalated"] is True


# --- HTTP ------------------------------------------------------------------------------------

def test_conversation_endpoints(retriever, tmp_path, monkeypatch):
    monkeypatch.delenv("DEMO_API_KEY", raising=False)
    llm = FakeLLM([text_response("Las licuadoras tienen 6 meses de garantía.")])
    app = create_app(service_factory=lambda: make_chat(llm, retriever, tmp_path))
    with TestClient(app) as client:
        sid = client.post("/chat", json={"message": "¿Garantía de una licuadora?"}).json()["session_id"]

        listing = client.get("/conversations").json()
        assert [c["id"] for c in listing] == [sid] and listing[0]["message_count"] == 2

        conv = client.get(f"/conversations/{sid}").json()
        assert conv["title"] == "¿Garantía de una licuadora?"
        assert conv["messages"][1]["content"].startswith("Las licuadoras")

        assert client.delete(f"/conversations/{sid}").status_code == 204
        assert client.get(f"/conversations/{sid}").status_code == 404
        assert client.delete(f"/conversations/{sid}").status_code == 404


def test_chat_rejects_unsafe_session_ids(retriever, tmp_path, monkeypatch):
    monkeypatch.delenv("DEMO_API_KEY", raising=False)
    app = create_app(service_factory=lambda: make_chat(FakeLLM([]), retriever, tmp_path))
    with TestClient(app) as client:
        r = client.post("/chat", json={"message": "hola", "session_id": "../../etc/passwd"})
        assert r.status_code == 422


def test_conversation_endpoints_require_api_key_when_configured(retriever, tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_API_KEY", "secreto")
    app = create_app(service_factory=lambda: make_chat(FakeLLM([]), retriever, tmp_path))
    with TestClient(app) as client:
        assert client.get("/conversations").status_code == 401
        assert client.get("/conversations", headers={"X-API-Key": "secreto"}).status_code == 200
