"""Caso de uso: consultar el estado de un pedido."""

import logging

from ..ports.orders import OrderRepositoryPort

logger = logging.getLogger(__name__)


class OrderStatusService:
    def __init__(self, repository: OrderRepositoryPort):
        self._repository = repository

    def consultar_estado_pedido(self, order_id: str) -> dict:
        """Consulta el estado de un pedido por su ID (p. ej. "ORD-1001").

        Nunca inventa datos: si el ID no existe retorna `encontrado=False` y
        `estado="no encontrado"`, sin producto ni fecha.
        """
        normalized = (order_id or "").strip().upper()
        order = self._repository.get(normalized) if normalized else None
        if order is None:
            logger.info("Pedido no encontrado: %r", order_id)
            return {
                "order_id": order_id,
                "encontrado": False,
                "estado": "no encontrado",
                "mensaje": f"No existe ningún pedido con el ID '{order_id}'.",
            }
        return {
            "order_id": order.order_id,
            "encontrado": True,
            "producto": order.producto,
            "estado": order.estado,
            "entrega_estimada": order.entrega_estimada,
        }
