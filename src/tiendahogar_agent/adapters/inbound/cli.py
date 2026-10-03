"""Adapter de entrada: chat interactivo en terminal."""

import uuid

from ...application.support_agent import SupportAgent


def run_cli(agent: SupportAgent) -> None:
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
