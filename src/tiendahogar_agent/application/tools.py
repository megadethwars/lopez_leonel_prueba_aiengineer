"""Tools que el LLM puede invocar.

`ToolRegistry` desacopla el grafo de las tools concretas: agregar una capacidad
nueva (p. ej. consultar inventario vía otro microservicio) es registrar su
esquema y su handler en `bootstrap.py`, sin tocar la orquestación.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable

from ..domain.models import HUMAN_SUPPORT_EMAIL, Escalation, EscalationCategory
from .order_status import OrderStatusService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolResult:
    output: dict[str, Any]
    escalation: Escalation | None = None


ToolHandler = Callable[[dict[str, Any]], ToolResult]


class ToolRegistry:
    def __init__(self) -> None:
        self._schemas: list[dict] = []
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, schema: dict, handler: ToolHandler) -> None:
        name = schema["name"]
        if name in self._handlers:
            raise ValueError(f"Tool duplicada: {name}")
        self._schemas.append(schema)
        self._handlers[name] = handler

    @property
    def schemas(self) -> list[dict]:
        return list(self._schemas)

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            logger.error("El LLM pidió una tool desconocida: %s", name)
            return ToolResult({"error": f"Herramienta desconocida: {name}"})
        return handler(args)


# Esquemas para la API de Claude (strict: argumentos siempre válidos).
CONSULTAR_ESTADO_PEDIDO_TOOL = {
    "name": "consultar_estado_pedido",
    "description": (
        "Consulta el estado, producto y entrega estimada de un pedido de TiendaHogar. "
        "Úsala siempre que el cliente pregunte por un pedido y proporcione su ID "
        "(formato ORD-XXXX). Si el cliente no da el ID, pídeselo en vez de llamar la tool."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "ID del pedido, p. ej. ORD-1001"},
        },
        "required": ["order_id"],
        "additionalProperties": False,
    },
}

ESCALAR_A_HUMANO_TOOL = {
    "name": "escalar_a_humano",
    "description": (
        "Remite el caso a un agente humano. Úsala SIEMPRE (en lugar de responder tú) cuando el "
        "cliente: pida un reembolso mayor a $500; se queje del trato de un empleado; tenga una "
        "disputa de facturación; o plantee cualquier tema legal."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "categoria": {"type": "string", "enum": [c.value for c in EscalationCategory]},
            "motivo": {"type": "string", "description": "Resumen breve del caso."},
        },
        "required": ["categoria", "motivo"],
        "additionalProperties": False,
    },
}


def _escalar_a_humano(args: dict[str, Any]) -> ToolResult:
    category = EscalationCategory(args.get("categoria", EscalationCategory.OTRO.value))
    escalation = Escalation(category, str(args.get("motivo", "")), "llm_tool")
    output = {
        "escalado": True,
        "canal": HUMAN_SUPPORT_EMAIL,
        "instruccion": "Informa al cliente que no puedes resolver este caso y que debe "
        f"escribir a {HUMAN_SUPPORT_EMAIL} para ser atendido por un humano.",
    }
    return ToolResult(output, escalation)


def build_tool_registry(order_service: OrderStatusService) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        CONSULTAR_ESTADO_PEDIDO_TOOL,
        lambda args: ToolResult(order_service.consultar_estado_pedido(str(args.get("order_id", "")))),
    )
    registry.register(ESCALAR_A_HUMANO_TOOL, _escalar_a_humano)
    return registry
