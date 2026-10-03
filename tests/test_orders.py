"""Tool consultar_estado_pedido: datos exactos para IDs válidos, sin inventar para inválidos."""

import inspect

import pytest

from tiendahogar_agent.application.order_status import OrderStatusService
from tiendahogar_agent.bootstrap import consultar_estado_pedido
from tiendahogar_agent.domain.models import Order


def test_tool_has_the_exact_required_signature():
    sig = inspect.signature(consultar_estado_pedido)
    assert list(sig.parameters) == ["order_id"]
    assert sig.parameters["order_id"].annotation is str
    assert sig.return_annotation is dict


def test_valid_order_returns_exact_table_row():
    assert consultar_estado_pedido("ORD-1001") == {
        "order_id": "ORD-1001",
        "encontrado": True,
        "producto": "Refrigeradora",
        "estado": "En tránsito",
        "entrega_estimada": "3 días hábiles",
    }


def test_delivered_order_has_no_estimated_date():
    result = consultar_estado_pedido("ORD-1002")
    assert result["estado"] == "Entregado"
    assert result["entrega_estimada"] is None


def test_order_id_is_normalized():
    assert consultar_estado_pedido("  ord-1003 ")["producto"] == "Lavadora"


@pytest.mark.parametrize("order_id", ["ORD-9999", "1001", "", "ORD-1001; DROP TABLE"])
def test_unknown_order_is_not_found_and_invents_nothing(order_id):
    result = consultar_estado_pedido(order_id)
    assert result["encontrado"] is False
    assert result["estado"] == "no encontrado"
    # No debe haber ningún dato de pedido fabricado.
    assert "producto" not in result
    assert "entrega_estimada" not in result


def test_service_works_with_any_repository_adapter():
    # Hexagonal: cualquier objeto con `get(order_id) -> Order | None` cumple el port.
    class OmsApiStub:
        def get(self, order_id: str) -> Order | None:
            return Order(order_id, "Estufa", "En tránsito", "2 días hábiles") if order_id == "ORD-7" else None

    service = OrderStatusService(OmsApiStub())
    assert service.consultar_estado_pedido("ord-7")["producto"] == "Estufa"
    assert service.consultar_estado_pedido("ORD-1001")["encontrado"] is False
