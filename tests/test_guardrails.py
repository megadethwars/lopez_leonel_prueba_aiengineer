"""Guardrail: detecta los casos que deben escalarse a un humano (Doc 4 y Doc 5)."""

import pytest

from tiendahogar_agent.guardrails import (
    EscalationCategory as C,
    check_input,
    check_output,
    escalation_message,
    extract_amounts,
)


@pytest.mark.parametrize(
    "message, category",
    [
        ("Quiero un reembolso de $800 por mi lavadora", C.REEMBOLSO_MAYOR_500),
        ("Necesito que me devuelvan mi dinero, fueron $1,250.00", C.REEMBOLSO_MAYOR_500),
        ("Quiero devolver mi refrigeradora de 2 mil dólares", C.REEMBOLSO_MAYOR_500),
        ("Un empleado me trató muy mal en la tienda", C.QUEJA_TRATO_EMPLEADO),
        ("Quiero poner una queja contra el vendedor, fue grosero", C.QUEJA_TRATO_EMPLEADO),
        ("Me cobraron dos veces en la factura", C.DISPUTA_FACTURACION),
        ("Hay un cargo en mi tarjeta que no reconozco", C.DISPUTA_FACTURACION),
        ("Voy a demandar a la tienda con mi abogado", C.TEMA_LEGAL),
        ("¿Esto es un tema legal? Iré a protección al consumidor", C.TEMA_LEGAL),
    ],
)
def test_guardrail_escalates(message, category):
    result = check_input(message)
    assert result is not None
    assert result.category == category


@pytest.mark.parametrize(
    "message",
    [
        "¿Cuánto dura la garantía de una licuadora?",
        "Quiero un reembolso de $300 por mi plancha",  # <= $500: el agente puede explicar la política
        "¿Cuándo me devuelven mi dinero del pedido ORD-1001?",  # el ID no es un monto
        "¿Puedo devolver un producto después de 30 días?",
        "¿Dónde está mi pedido ORD-1003?",
    ],
)
def test_guardrail_does_not_fire_on_normal_questions(message):
    assert check_input(message) is None


def test_amount_extraction_ignores_order_ids_and_durations():
    assert extract_amounts("ord-1001 hace 600 dias, 2-3 dias, monto $1.200,50") == [1200.50]


def test_escalation_message_points_to_human_channel():
    assert "soporte@tiendahogar.example" in escalation_message(C.REEMBOLSO_MAYOR_500)


def test_output_guardrail_blocks_refund_approval():
    assert check_output("¡Listo! Tu reembolso ha sido aprobado.") is not None
    assert check_output("Los reembolsos se procesan en 5-10 días hábiles.") is None
