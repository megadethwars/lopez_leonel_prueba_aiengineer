"""Chat interactivo en terminal:  python -m tiendahogar_agent.cli  (con PYTHONPATH=src)."""

import uuid

from .llm import AnthropicLLM
from .retriever import get_retriever
from .graph import SupportAgent


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
