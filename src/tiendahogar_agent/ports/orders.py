"""Port del repositorio de pedidos.

Adapters posibles: tabla mock en memoria (actual), API del OMS detrás de Apigee,
una base de datos SQL, etc.
"""

from typing import Protocol

from ..domain.models import Order


class OrderRepositoryPort(Protocol):
    def get(self, order_id: str) -> Order | None:
        """El pedido con ese ID (ya normalizado), o None si no existe."""
        ...
