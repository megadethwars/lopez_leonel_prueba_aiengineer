"""Caso de uso principal: un turno de conversación del agente de soporte (LangGraph).

    START → input_guardrail ─┬─(escalar)──────────────────────────────→ escalate → END
                             └→ retrieve ─┬─(sin contexto)─────────────→ no_context → END
                                          └→ agent ⇄ tools
                                               └─(fin)→ output_guardrail → END

Solo depende de `domain` y de `ports`: el LLM, el retriever, las tools, la skill
y el notificador se inyectan desde `bootstrap.py`.

El historial `messages` está en formato de la API de Claude y es append-only
(nunca se reescriben turnos anteriores), requisito para conservar los bloques de
thinking entre turnos y aprovechar el prompt cache.
"""

import json
import logging
import operator
import re
import time
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from ..domain.guardrails import check_input, check_output, escalation_message
from ..domain.models import HUMAN_SUPPORT_EMAIL, Escalation
from ..domain.text import normalize
from ..logging_config import preview, session_id_var
from ..ports.llm import LLMPort
from ..ports.notifications import EscalationNotifierPort
from ..ports.retrieval import RetrieverPort
from .prompts import NO_CONTEXT_MESSAGE, user_turn
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

_ORDER_INTENT = re.compile(r"\bord-?\d+\b|\bpedido|\borden\b")
# Saludos, agradecimientos y despedidas: no necesitan documentos, pero sí una respuesta
# cordial del LLM (no la plantilla de "sin información").
_SMALL_TALK = re.compile(
    r"^\W*(hola|holi|hey|buen[oa]s?( (dias|tardes|noches))?|saludos|que tal|como estas|"
    r"muchas gracias|gracias|mil gracias|te agradezco|ok|vale|perfecto|genial|excelente|"
    r"adios|hasta luego|hasta pronto|nos vemos|chao|bye)\b"
)
_SMALL_TALK_MAX_WORDS = 8


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]  # historial persistente (append-only)
    question: str
    sources: list[dict]
    escalation: dict | None  # Escalation.to_dict(): el estado debe ser serializable
    tool_calls: list[dict]
    tool_iterations: int
    route: str
    answer: str


def _text_of(content: list[dict]) -> str:
    return "\n".join(b["text"] for b in content if b.get("type") == "text").strip()


def _assistant_text(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def build_graph(
    llm: LLMPort,
    retriever: RetrieverPort,
    tools: ToolRegistry,
    system_prompt: str,
    checkpointer: Any | None = None,
    max_tool_iterations: int = 4,
):
    def input_guardrail(state: AgentState) -> AgentState:
        hit = check_input(state["question"])
        if hit:
            logger.warning("Guardrail de entrada activado: categoria=%s motivo=%s",
                           hit.category.value, hit.reason)
            escalation = Escalation(hit.category, hit.reason, "input_guardrail")
            return {"escalation": escalation.to_dict(), "route": "escalate"}
        logger.info("Guardrail de entrada: sin coincidencias")
        return {"escalation": None, "route": "retrieve"}

    def retrieve(state: AgentState) -> AgentState:
        question = state["question"]
        started = time.perf_counter()
        sources = [
            {"doc_id": r.document.doc_id, "title": r.document.title,
             "content": r.document.content, "score": round(r.score, 3)}
            for r in retriever.retrieve(question)
        ]
        logger.info(
            "RAG: %d documento(s) relevantes en %.0f ms: %s",
            len(sources), (time.perf_counter() - started) * 1000,
            ", ".join(f"{s['doc_id']}={s['score']}" for s in sources) or "ninguno",
        )
        has_history = bool(state.get("messages"))
        normalized = normalize(question)
        small_talk = (bool(_SMALL_TALK.search(normalized))
                      and len(normalized.split()) <= _SMALL_TALK_MAX_WORDS)
        if small_talk and not sources:
            logger.info("Mensaje conversacional (saludo/agradecimiento/despedida) → LLM sin contexto")
        if not sources and not has_history and not small_talk and not _ORDER_INTENT.search(normalized):
            logger.info("RAG: sin contexto ni intención de pedido → respuesta 'sin información'")
            return {"sources": [], "route": "no_context"}
        return {"sources": sources, "route": "agent", "messages": [user_turn(question, sources)]}

    def escalate(state: AgentState) -> AgentState:
        escalation = Escalation.from_dict(state["escalation"])
        answer = escalation_message(escalation.category)
        logger.info("Escalamiento a humano sin llamar al LLM: categoria=%s", escalation.category.value)
        return {
            "answer": answer,
            "messages": [user_turn(state["question"], []), _assistant_text(answer)],
        }

    def no_context(state: AgentState) -> AgentState:
        return {
            "answer": NO_CONTEXT_MESSAGE,
            "messages": [user_turn(state["question"], []), _assistant_text(NO_CONTEXT_MESSAGE)],
        }

    def agent(state: AgentState) -> AgentState:
        response = llm.create(system=system_prompt, messages=state["messages"], tools=tools.schemas)
        content = response["content"] or [{"type": "text", "text": NO_CONTEXT_MESSAGE}]
        stop = response["stop_reason"]
        if stop == "refusal":
            logger.warning("El LLM rechazó la petición (refusal); se responde con mensaje seguro")
            content = [{"type": "text", "text": NO_CONTEXT_MESSAGE}]
        elif stop == "max_tokens":
            logger.warning("Respuesta del LLM truncada por max_tokens")
        route = "tools" if stop == "tool_use" else "output_guardrail"
        logger.info("Agente: stop_reason=%s → %s", stop, route)
        return {"messages": [{"role": "assistant", "content": content}], "route": route}

    def run_tools(state: AgentState) -> AgentState:
        last = state["messages"][-1]
        results, calls, escalation = [], list(state.get("tool_calls") or []), state.get("escalation")
        for block in last["content"]:
            if block.get("type") != "tool_use":
                continue
            name, args = block["name"], block.get("input") or {}
            started = time.perf_counter()
            result = tools.execute(name, args)
            if result.escalation:
                escalation = result.escalation.to_dict()
            logger.info("Tool %s input=%s output=%s (%.1f ms)", name,
                        json.dumps(args, ensure_ascii=False),
                        json.dumps(result.output, ensure_ascii=False),
                        (time.perf_counter() - started) * 1000)
            calls.append({"tool": name, "input": args, "output": result.output})
            results.append({"type": "tool_result", "tool_use_id": block["id"],
                            "content": json.dumps(result.output, ensure_ascii=False)})

        iterations = (state.get("tool_iterations") or 0) + 1
        new_messages: list[dict] = [{"role": "user", "content": results}]
        route = "agent"
        if iterations >= max_tool_iterations:
            # Corta bucles de tools: cierra el turno con un mensaje seguro.
            logger.warning("Límite de %d iteraciones de tools alcanzado; se corta el bucle",
                           max_tool_iterations)
            new_messages.append(_assistant_text(NO_CONTEXT_MESSAGE))
            route = "output_guardrail"
        return {"messages": new_messages, "tool_calls": calls, "tool_iterations": iterations,
                "escalation": escalation, "route": route}

    def output_guardrail(state: AgentState) -> AgentState:
        answer = _text_of(state["messages"][-1]["content"]) or NO_CONTEXT_MESSAGE
        escalation = state.get("escalation")
        hit = check_output(answer)
        if hit:
            logger.warning("Guardrail de salida activado: %s Respuesta reemplazada.", hit.reason)
            escalation = Escalation(hit.category, hit.reason, "output_guardrail").to_dict()
            answer = escalation_message(hit.category)
        elif escalation and HUMAN_SUPPORT_EMAIL not in answer:
            logger.warning("Escalamiento sin canal humano en la respuesta; se usa la plantilla")
            answer = escalation_message(Escalation.from_dict(escalation).category)
        return {"answer": answer, "escalation": escalation}

    def by_route(state: AgentState) -> str:
        return state["route"]

    graph = StateGraph(AgentState)
    for name, fn in [("input_guardrail", input_guardrail), ("retrieve", retrieve),
                     ("escalate", escalate), ("no_context", no_context), ("agent", agent),
                     ("tools", run_tools), ("output_guardrail", output_guardrail)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "input_guardrail")
    graph.add_conditional_edges("input_guardrail", by_route, ["escalate", "retrieve"])
    graph.add_conditional_edges("retrieve", by_route, ["no_context", "agent"])
    graph.add_conditional_edges("agent", by_route, ["tools", "output_guardrail"])
    graph.add_conditional_edges("tools", by_route, ["agent", "output_guardrail"])
    for terminal in ("escalate", "no_context", "output_guardrail"):
        graph.add_edge(terminal, END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


class SupportAgent:
    """Fachada del caso de uso: una llamada = un turno de conversación."""

    def __init__(
        self,
        llm: LLMPort,
        retriever: RetrieverPort,
        tools: ToolRegistry,
        system_prompt: str,
        notifier: EscalationNotifierPort,
        checkpointer: Any | None = None,
        max_tool_iterations: int = 4,
    ):
        self.graph = build_graph(llm, retriever, tools, system_prompt, checkpointer,
                                 max_tool_iterations)
        self._notifier = notifier

    def has_memory(self, session_id: str) -> bool:
        """True si el checkpointer conserva el historial de esta sesión."""
        state = self.graph.get_state({"configurable": {"thread_id": session_id}})
        return bool(state.values.get("messages"))

    def ask(
        self,
        question: str,
        session_id: str,
        prior_turns: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Procesa un turno. `prior_turns` (pregunta, respuesta) se usa solo si el agente no
        recuerda la sesión (p. ej. tras un reinicio) para restaurar el contexto."""
        turn_input: AgentState = {
            "question": question, "sources": [], "escalation": None,
            "tool_calls": [], "tool_iterations": 0, "answer": "",
        }
        token = session_id_var.set(session_id)  # todos los logs del turno llevan el session_id
        started = time.perf_counter()
        try:
            logger.info("Pregunta recibida: %r", preview(question))
            if prior_turns and not self.has_memory(session_id):
                logger.info("Restaurando contexto de %d turno(s) previos desde el historial",
                            len(prior_turns))
                # Sin bloque <contexto>: marcarlo como "sin documentos" haría creer al modelo
                # que antes respondió sin fuentes.
                turn_input["messages"] = [
                    msg for q, a in prior_turns
                    for msg in ({"role": "user", "content": f"<pregunta_cliente>\n{q}\n</pregunta_cliente>"},
                                _assistant_text(a))
                ]
            state = self.graph.invoke(turn_input, config={"configurable": {"thread_id": session_id}})
            escalation = state.get("escalation")
            if escalation:
                self._notify(Escalation.from_dict(escalation), session_id, question)
            logger.info(
                "Turno completado en %.0f ms: escalado=%s fuentes=%d tools=%d",
                (time.perf_counter() - started) * 1000, escalation is not None,
                len(state.get("sources") or []), len(state.get("tool_calls") or []),
            )
        except Exception:
            logger.exception("Error procesando el turno")
            raise
        finally:
            session_id_var.reset(token)
        return {
            "session_id": session_id,
            "answer": state["answer"],
            "escalated": escalation is not None,
            "escalation": escalation,
            "sources": [{k: s[k] for k in ("doc_id", "title", "score")}
                        for s in state.get("sources") or []],
            "tool_calls": state.get("tool_calls") or [],
        }

    def _notify(self, escalation: Escalation, session_id: str, question: str) -> None:
        # Un fallo al notificar (p. ej. broker caído) no debe dejar al cliente sin respuesta.
        try:
            self._notifier.notify(escalation, session_id, question)
        except Exception:
            logger.exception("No se pudo notificar el escalamiento; la respuesta continúa")
