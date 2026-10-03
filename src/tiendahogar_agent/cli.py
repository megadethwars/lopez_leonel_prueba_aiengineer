"""Chat interactivo en terminal:  python -m tiendahogar_agent.cli  (con PYTHONPATH=src)."""

import os
import uuid

from .logging_config import setup_logging

# En la terminal los logs ensucian la conversación: por defecto solo WARNING+ (LOG_LEVEL=INFO para verlos).
setup_logging(os.getenv("LOG_LEVEL") or "WARNING")

from .graph import SupportAgent  # noqa: E402
from .llm import AnthropicLLM  # noqa: E402
from .retriever import get_retriever  # noqa: E402


def main() -> None:
    agent = SupportAgent(llm=AnthropicLLM(), retriever=get_retriever())
    session_id = str(uuid.uuid4())
    print("TiendaHogar — asistente de soporte. Escribe 'salir' para terminar.\n")
    while True:
        try:
            question = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"salir", "exit", "quit"}:
            break
        if not question:
            continue
        result = agent.ask(question, session_id=session_id)
        print(f"\nAsistente: {result['answer']}")
        if result["sources"]:
            print("  fuentes:", ", ".join(f"{s['title']} ({s['score']})" for s in result["sources"]))
        for call in result["tool_calls"]:
            print(f"  tool: {call['tool']}({call['input']}) -> {call['output']}")
        if result["escalated"]:
            print(f"  escalado: {result['escalation']}")
        print()


if __name__ == "__main__":
    main()
