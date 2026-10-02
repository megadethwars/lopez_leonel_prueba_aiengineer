"""Contrato HTTP del microservicio (agente con LLM falso)."""

from fastapi.testclient import TestClient

from conftest import FakeLLM
from tiendahogar_agent.api import create_app
from tiendahogar_agent.graph import SupportAgent


def test_chat_endpoint_escalates_and_returns_session(retriever, monkeypatch):
    monkeypatch.delenv("DEMO_API_KEY", raising=False)
    app = create_app(agent=SupportAgent(FakeLLM([]), retriever))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        r = client.post("/chat", json={"message": "Voy a demandar a la tienda"})
        assert r.status_code == 200
        body = r.json()
        assert body["escalated"] is True and body["session_id"]


def test_chat_requires_api_key_when_configured(retriever, monkeypatch):
    monkeypatch.setenv("DEMO_API_KEY", "secreto")
    app = create_app(agent=SupportAgent(FakeLLM([]), retriever))
    with TestClient(app) as client:
        assert client.post("/chat", json={"message": "hola"}).status_code == 401
        ok = client.post("/chat", json={"message": "Voy a demandar a la tienda"},
                         headers={"X-API-Key": "secreto"})
        assert ok.status_code == 200
