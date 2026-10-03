"""Adapter de `OrderRepositoryPort` con la tabla mock entregada por el cliente."""

from ....domain.models import Order

# "—" en la tabla original = sin entrega estimada (None).
MOCK_ORDERS: list[Order] = [
    Order("ORD-1001", "Refrigeradora", "En tránsito", "3 días hábiles"),
    Order("ORD-1002", "Licuadora", "Entregado", None),
    Order("ORD-1003", "Lavadora", "Procesando", "6 días hábiles"),
    Order("ORD-1004", "Tostadora", "Cancelado", None),
]


class InMemoryOrderRepository:
    def __init__(self, orders: list[Order] | None = None):
        self._orders = {o.order_id: o for o in (MOCK_ORDERS if orders is None else orders)}

    def get(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)
