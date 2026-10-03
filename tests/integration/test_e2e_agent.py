"""Pruebas end-to-end: HTTP real → agente real → Claude real, evaluadas por un juez LLM.

Cada caso pasa solo si:
1. las verificaciones deterministas sobre el JSON de /chat se cumplen (escalamiento,
   tools, fuentes), y
2. el juez aprueba todos los criterios de la rúbrica (incluida la fidelidad a la base
   de conocimiento).
"""

import time

import pytest

from integration.cases import CASES, EvalCase

pytestmark = pytest.mark.integration


def _converse(client, turns: list[str]) -> tuple[list[tuple[str, str]], dict, float]:
    """Envía los turnos en una misma sesión; devuelve historial previo, última respuesta y latencia."""
    session_id, history, last, latency = None, [], {}, 0.0
    for i, message in enumerate(turns):
        started = time.perf_counter()
        r = client.post("/chat", json={"message": message, "session_id": session_id})
        latency = time.perf_counter() - started
        assert r.status_code == 200, f"/chat respondió {r.status_code}: {r.text}"
        last = r.json()
        session_id = last["session_id"]
        if i < len(turns) - 1:
            history.append((message, last["answer"]))
    return history, last, latency


def _deterministic_checks(case: EvalCase, body: dict) -> list[str]:
    failures = []
    tools = [t["tool"] for t in body["tool_calls"]]
    sources = [s["doc_id"] for s in body["sources"]]
    if case.expect_escalated is not None and body["escalated"] != case.expect_escalated:
        failures.append(f"escalated={body['escalated']} (esperado {case.expect_escalated})")
    if case.expect_escalation_source and (body["escalation"] or {}).get("source") != case.expect_escalation_source:
        failures.append(f"escalation.source={(body['escalation'] or {}).get('source')} "
                        f"(esperado {case.expect_escalation_source})")
    if case.expect_tool and case.expect_tool not in tools:
        failures.append(f"no llamó a {case.expect_tool} (tools={tools})")
    if case.expect_no_tools and tools:
        failures.append(f"llamó tools inesperadas: {tools}")
    if case.expect_source and case.expect_source not in sources:
        failures.append(f"falta la fuente {case.expect_source} (fuentes={sources})")
    return failures


def _params():
    for c in CASES:
        marks = [pytest.mark.xfail(reason=c.known_issue, strict=False)] if c.known_issue else []
        yield pytest.param(c, id=c.id, marks=marks)


@pytest.mark.parametrize("case", list(_params()))
def test_agent_end_to_end(case: EvalCase, client, judge, record):
    history, body, latency = _converse(client, case.turns)
    question, answer = case.turns[-1], body["answer"]

    check_failures = _deterministic_checks(case, body)
    trace = {"escalated": body["escalated"], "escalation": body["escalation"],
             "fuentes": [s["title"] for s in body["sources"]], "tools": body["tool_calls"]}
    result = judge.evaluate(question, answer, case.criteria, history=history, agent_trace=trace)

    passed = not check_failures and result.passed
    record({
        "case": case.id, "tags": case.tags, "known_issue": case.known_issue, "question": question, "answer": answer,
        "latency_s": round(latency, 2), "passed": passed,
        "check_failures": check_failures,
        "judge_passed": result.passed,
        "judge_failures": [f.criterion[:80] for f in result.failures],
        "judge_criteria": [c.model_dump() for c in result.verdict.criteria],
        "judge_summary": result.verdict.summary, "judge_model": result.model,
        "judge_tokens_in": result.input_tokens, "judge_tokens_out": result.output_tokens,
    })

    assert not check_failures, f"Verificaciones deterministas fallidas: {check_failures}\nRespuesta: {answer}"
    assert result.passed, (
        "El juez reprobó la respuesta:\n"
        + "\n".join(f"- {f.criterion}: {f.reasoning}" for f in result.failures)
        + f"\nRespuesta: {answer}"
    )
