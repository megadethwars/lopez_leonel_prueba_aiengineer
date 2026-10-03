"""Entidades del dominio de soporte de TiendaHogar."""

from dataclasses import dataclass
from enum import Enum

# Doc 5: canal humano para los casos que el asistente no debe resolver.
HUMAN_SUPPORT_EMAIL = "soporte@tiendahogar.example"


@dataclass(frozen=True)
class Document:
    """Documento de la base de conocimiento (política)."""

    doc_id: str
    title: str
    content: str


@dataclass(frozen=True)
class RetrievedDocument:
    """Documento recuperado para una consulta, con su score de relevancia."""

    document: Document
    score: float


@dataclass(frozen=True)
class Order:
    """Pedido de un cliente. `entrega_estimada` es None cuando no aplica."""

    order_id: str
    producto: str
    estado: str
    entrega_estimada: str | None


class EscalationCategory(str, Enum):
    REEMBOLSO_MAYOR_500 = "reembolso_mayor_500"
    QUEJA_TRATO_EMPLEADO = "queja_trato_empleado"
    DISPUTA_FACTURACION = "disputa_facturacion"
    TEMA_LEGAL = "tema_legal"
    OTRO = "otro"


@dataclass(frozen=True)
class Escalation:
    """Caso remitido a un agente humano.

    source: quién lo detectó — "input_guardrail", "llm_tool" u "output_guardrail".
    """

    category: EscalationCategory
    reason: str
    source: str

    def to_dict(self) -> dict:
        return {"category": self.category.value, "reason": self.reason, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict) -> "Escalation":
        return cls(EscalationCategory(data["category"]), data["reason"], data["source"])
