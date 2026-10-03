"""Guardrails deterministas (capa 1) para los casos que el agente NO debe resolver.

Fuente: Doc 4 (reembolsos > $500 requieren supervisor humano) y Doc 5 (quejas de
trato, disputas de facturación y temas legales van a soporte humano).

Capa 1 (este módulo): reglas regex sobre el mensaje del cliente, antes del LLM.
Barato, auditable y testeable; si se activa, el LLM ni siquiera se invoca.
Capa 2: la tool `escalar_a_humano` + system prompt, para casos que las reglas
no capturan (paráfrasis, números escritos en letras, etc.).
Capa 3 (salida): `check_output` impide que una respuesta "apruebe" un reembolso.
"""

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

from .knowledge_base import HUMAN_SUPPORT_EMAIL

REFUND_LIMIT = 500.0


class EscalationCategory(str, Enum):
    REEMBOLSO_MAYOR_500 = "reembolso_mayor_500"
    QUEJA_TRATO_EMPLEADO = "queja_trato_empleado"
    DISPUTA_FACTURACION = "disputa_facturacion"
    TEMA_LEGAL = "tema_legal"
    OTRO = "otro"


@dataclass(frozen=True)
class GuardrailResult:
    category: EscalationCategory
    reason: str


def normalize(text: str) -> str:
    """Minúsculas y sin tildes, para que las reglas no dependan de la ortografía."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


_LEGAL = re.compile(
    r"abogad|\bdemand(a|ar|are|aremos|e)\b|juicio|tribunal|juzgado|litigio|"
    r"(accion|acciones|proceso|tema|asunto|via|reclamo) legal|legalmente|denuncia formal|"
    r"profeco|diaco|proteccion al consumidor|ministerio publico|indemniz"
)
_BILLING_SUBJECT = re.compile(r"factur|cobr|cargo|tarjeta|estado de cuenta")
_BILLING_DISPUTE = re.compile(
    r"disput|incorrect|doble|dos veces|duplicad|no reconozco|indebid|de mas|error|"
    r"equivocad|no autoric|no corresponde|fraud|extra"
)
_CHARGEBACK = re.compile(r"contracargo|chargeback")
_EMPLOYEE_ACTOR = re.compile(
    r"emplead|vendedor|cajer|personal|repartidor|tecnico|trabajador|asesor|dependient|"
    r"instalador|gerente|encargad|me atendi|atencion"
)
_EMPLOYEE_COMPLAINT = re.compile(
    r"queja|quejar|reclam|denunci|groser|maltrat|mal trato|irrespetuos|ofend|insult|grit|"
    r"discrimin|prepotent|descortes|humill|mala atencion|me trat\w* (muy )?mal|pesim"
)
_EMPLOYEE_STRONG = re.compile(
    r"me trat\w* (muy )?mal|mal trato|maltrat|mala atencion|me (gritaron|insultaron|humillaron)"
)
_REFUND = re.compile(
    r"reembols|refund|devolu|devolv|devuelv|regres\w* (mi |el )?dinero|mi dinero de vuelta"
)
_ORDER_ID = re.compile(r"\bord-?\d+\b")
# Montos: "$1,200", "1.200", "750.50", "2 mil". Se excluyen cantidades de tiempo/porcentaje.
_AMOUNT = re.compile(
    r"(?<![\d.,])(\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,](\d{1,2}))?(?!\d|[.,]\d)(\s*mil\b)?"
    r"(?!\s*(?:dias?|meses?|mes|semanas?|anos?|horas?|%|-))"
)


def extract_amounts(text: str) -> list[float]:
    """Extrae montos numéricos de un texto ya normalizado (ignora IDs de pedido)."""
    cleaned = _ORDER_ID.sub(" ", text)
    amounts = []
    for integer, decimals, mil in _AMOUNT.findall(cleaned):
        value = float(re.sub(r"[.,]", "", integer) + (f".{decimals}" if decimals else ""))
        if mil:
            value *= 1000
        amounts.append(value)
    return amounts


def check_input(message: str) -> GuardrailResult | None:
    """Retorna la categoría de escalamiento si el mensaje debe ir a un humano."""
    text = normalize(message)

    if _LEGAL.search(text):
        return GuardrailResult(EscalationCategory.TEMA_LEGAL, "Menciona un tema legal.")
    if _CHARGEBACK.search(text) or (
        _BILLING_SUBJECT.search(text) and _BILLING_DISPUTE.search(text)
    ):
        return GuardrailResult(
            EscalationCategory.DISPUTA_FACTURACION, "Disputa sobre un cobro o factura."
        )
    if _EMPLOYEE_STRONG.search(text) or (
        _EMPLOYEE_ACTOR.search(text) and _EMPLOYEE_COMPLAINT.search(text)
    ):
        return GuardrailResult(
            EscalationCategory.QUEJA_TRATO_EMPLEADO, "Queja sobre el trato de un empleado."
        )
    if _REFUND.search(text):
        over = [a for a in extract_amounts(text) if a > REFUND_LIMIT]
        if over:
            return GuardrailResult(
                EscalationCategory.REEMBOLSO_MAYOR_500,
                f"Reembolso/devolución por ${max(over):,.2f} (> ${REFUND_LIMIT:,.0f}).",
            )
    return None


_ESCALATION_MESSAGES = {
    EscalationCategory.REEMBOLSO_MAYOR_500: (
        "Según nuestra política, los reembolsos mayores a $500 requieren la aprobación de un "
        "supervisor humano, por lo que no puedo aprobar ni gestionar este reembolso."
    ),
    EscalationCategory.QUEJA_TRATO_EMPLEADO: (
        "Lamento mucho lo ocurrido. Las quejas sobre el trato de un empleado son atendidas "
        "directamente por nuestro equipo humano de soporte."
    ),
    EscalationCategory.DISPUTA_FACTURACION: (
        "Las disputas de facturación son atendidas directamente por nuestro equipo humano de "
        "soporte, por lo que no puedo resolver este caso."
    ),
    EscalationCategory.TEMA_LEGAL: (
        "Los temas legales son atendidos exclusivamente por nuestro equipo humano de soporte, "
        "por lo que no puedo resolver este caso."
    ),
    EscalationCategory.OTRO: "Este caso debe ser atendido por nuestro equipo humano de soporte.",
}


def escalation_message(category: EscalationCategory) -> str:
    return (
        f"{_ESCALATION_MESSAGES[category]} Por favor escribe a {HUMAN_SUPPORT_EMAIL} "
        "y un agente humano te ayudará."
    )


_REFUND_APPROVAL = re.compile(
    r"(he|hemos|queda|ha sido|fue|esta) (aprobad|autorizad)\w*.{0,40}reembolso|"
    r"reembolso.{0,40}(ha sido|fue|queda|esta) (aprobad|autorizad)|"
    r"(apruebo|autorizo) (tu|el|su) reembolso"
)


def check_output(answer: str) -> GuardrailResult | None:
    """Detecta respuestas que aprueban un reembolso (el agente nunca debe hacerlo)."""
    if _REFUND_APPROVAL.search(normalize(answer)):
        return GuardrailResult(
            EscalationCategory.REEMBOLSO_MAYOR_500, "La respuesta intentaba aprobar un reembolso."
        )
    return None


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
