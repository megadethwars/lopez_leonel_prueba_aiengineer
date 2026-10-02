"""Tool consultar_estado_pedido: datos exactos para IDs válidos, sin inventar para inválidos."""

import pytest

from tiendahogar_agent.orders import consultar_estado_pedido


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
