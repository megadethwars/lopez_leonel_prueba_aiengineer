"""Fixtures de las pruebas de integración (agente real + Claude real + juez LLM).

Solo corren con `pytest -m integration` y si hay ANTHROPIC_API_KEY. Consumen tokens.
Al terminar escriben un reporte en reports/llm_eval_report.{json,md}.
"""

import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from integration.judge import ClaudeJudge
from tiendahogar_agent.adapters.inbound.http.api import create_app
from tiendahogar_agent.bootstrap import build_support_agent
from tiendahogar_agent.config import settings

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"
_RESULTS: list[dict] = []


def pytest_collection_modifyitems(config, items):
    if os.getenv("ANTHROPIC_API_KEY"):
        return
    skip = pytest.mark.skip(reason="Requiere ANTHROPIC_API_KEY (pruebas de integración con Claude real)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def client():
    """Microservicio real en proceso: HTTP → FastAPI → LangGraph → Claude + RAG + tools."""
    os.environ.pop("DEMO_API_KEY", None)
    with TestClient(create_app(agent_factory=build_support_agent)) as c:
        yield c


@pytest.fixture(scope="session")
def judge():
    j = ClaudeJudge()
    assert j.model != settings.anthropic_model, (
        "El juez debe ser un modelo distinto al del agente (evita la auto-preferencia). "
        "Cambia JUDGE_MODEL o ANTHROPIC_MODEL."
    )
    return j


@pytest.fixture(scope="session")
def record():
    """Acumula resultados de cada caso para el reporte final."""
    def _record(entry: dict) -> None:
        _RESULTS.append(entry)
    return _record


def pytest_sessionfinish(session, exitstatus):
    if not _RESULTS:
        return
    REPORT_DIR.mkdir(exist_ok=True)
    passed = sum(r["passed"] for r in _RESULTS)
    judge_in = sum(r.get("judge_tokens_in", 0) for r in _RESULTS)
    judge_out = sum(r.get("judge_tokens_out", 0) for r in _RESULTS)
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "agent_model": settings.anthropic_model,
        "judge_model": _RESULTS[0].get("judge_model"),
        "cases": len(_RESULTS),
        "passed": passed,
        "pass_rate": round(passed / len(_RESULTS), 3),
        "judge_tokens": {"input": judge_in, "output": judge_out},
        "results": _RESULTS,
    }
    (REPORT_DIR / "llm_eval_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Reporte de evaluación end-to-end",
        "",
        f"- Fecha: {summary['timestamp']}",
        f"- Modelo del agente: `{summary['agent_model']}` · Juez: `{summary['judge_model']}`",
        f"- Resultado: **{passed}/{len(_RESULTS)}** casos aprobados ({summary['pass_rate']:.0%})",
        f"- Tokens del juez: {judge_in} entrada / {judge_out} salida",
        "",
        "| Caso | Resultado | Verificaciones | Juez | Latencia |",
        "|---|---|---|---|---|",
    ]
    for r in _RESULTS:
        checks = "ok" if not r["check_failures"] else "; ".join(r["check_failures"])
        judge_txt = "ok" if r["judge_passed"] else "; ".join(r["judge_failures"])
        status = "✅" if r["passed"] else ("⚠️ conocido" if r.get("known_issue") else "❌")
        lines.append(f"| {r['case']} | {status} | {checks} | {judge_txt} | {r['latency_s']} s |")
    lines += ["", "## Detalle", ""]
    for r in _RESULTS:
        lines += [f"### {r['case']} {'✅' if r['passed'] else '❌'}", "",
                  f"**Cliente:** {r['question']}", "", f"**Asistente:** {r['answer']}", ""]
        for c in r.get("judge_criteria", []):
            lines.append(f"- {'✅' if c['passed'] else '❌'} {c['criterion']} — _{c['reasoning']}_")
        lines.append("")
    (REPORT_DIR / "llm_eval_report.md").write_text("\n".join(lines), encoding="utf-8")
