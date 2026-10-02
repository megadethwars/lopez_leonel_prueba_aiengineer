"""Tool `consultar_estado_pedido` sobre una tabla mock de pedidos."""

from typing import Any

# Tabla mock entregada por el cliente. "—" en el original = sin entrega estimada (None).
_ORDERS: dict[str, dict[str, Any]] = {
    "ORD-1001": {"producto": "Refrigeradora", "estado": "En tránsito", "entrega_estimada": "3 días hábiles"},
    "ORD-1002": {"producto": "Licuadora", "estado": "Entregado", "entrega_estimada": None},
    "ORD-1003": {"producto": "Lavadora", "estado": "Procesando", "entrega_estimada": "6 días hábiles"},
    "ORD-1004": {"producto": "Tostadora", "estado": "Cancelado", "entrega_estimada": None},
}


def consultar_estado_pedido(order_id: str) -> dict:
    """Consulta el estado de un pedido por su ID (p. ej. "ORD-1001").

    Nunca inventa datos: si el ID no existe retorna `encontrado=False` y
    `estado="no encontrado"`, sin producto ni fecha.
    """
    normalized = (order_id or "").strip().upper()
    order = _ORDERS.get(normalized)
    if order is None:
        return {
            "order_id": order_id,
            "encontrado": False,
            "estado": "no encontrado",
            "mensaje": f"No existe ningún pedido con el ID '{order_id}'.",
        }
    return {"order_id": normalized, "encontrado": True, **order}


# Esquema de la tool para la API de Claude (strict: argumentos siempre válidos).
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
