"""Reglas de dependencia de la arquitectura hexagonal (verificadas sobre el código fuente).

    domain       → no depende de nada del proyecto ni de frameworks
    ports        → solo de domain
    application  → solo de domain, ports (y LangGraph como motor de orquestación)
    adapters     → pueden depender de todo lo anterior, nunca entre adapters de entrada y salida
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "tiendahogar_agent"
FRAMEWORKS = {"anthropic", "fastapi", "fastembed", "numpy", "uvicorn", "pydantic", "dotenv"}


def _imports(layer: str) -> set[tuple[str, str]]:
    """(archivo, módulo importado) para cada import de la capa, resolviendo imports relativos."""
    found = set()
    for path in (PACKAGE / layer).rglob("*.py"):
        package_parts = ["tiendahogar_agent", *path.relative_to(PACKAGE).parent.parts]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found |= {(path.name, alias.name) for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package_parts[: len(package_parts) - node.level + 1]
                    module = ".".join(base + ([node.module] if node.module else []))
                else:
                    module = node.module or ""
                found.add((path.name, module))
    return found


def _violations(layer: str, forbidden_internal: set[str], forbid_frameworks: bool = True):
    bad = []
    for file, module in _imports(layer):
        top = module.split(".")[0]
        if forbid_frameworks and top in FRAMEWORKS:
            bad.append(f"{layer}/{file} importa {module}")
        if top == "tiendahogar_agent":
            sub = module.split(".")[1] if "." in module else ""
            if sub in forbidden_internal:
                bad.append(f"{layer}/{file} importa {module}")
    return bad


@pytest.mark.parametrize(
    "layer, forbidden",
    [
        ("domain", {"ports", "application", "adapters", "bootstrap", "config", "main", "logging_config"}),
        ("ports", {"application", "adapters", "bootstrap", "config", "main"}),
        ("application", {"adapters", "bootstrap", "config", "main"}),
    ],
)
def test_core_layers_respect_dependency_rule(layer, forbidden):
    assert _violations(layer, forbidden) == []


def test_inbound_adapters_do_not_depend_on_outbound_adapters():
    bad = [f"inbound/{f} importa {m}" for f, m in _imports("adapters/inbound")
           if m.startswith("tiendahogar_agent.adapters.outbound") or m.split(".")[0] in {"anthropic", "fastembed"}]
    assert bad == []
