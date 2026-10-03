"""Port de notificación de escalamientos.

Adapters posibles: log (actual), un tópico de Kafka `support.escalation.created`,
un webhook al CRM o sistema de ticketing, email, etc.
"""

from typing import Protocol

from ..domain.models import Escalation


class EscalationNotifierPort(Protocol):
    def notify(self, escalation: Escalation, session_id: str, question: str) -> None:
        """Publica que un caso fue remitido a un humano. No debe bloquear la respuesta."""
        ...
