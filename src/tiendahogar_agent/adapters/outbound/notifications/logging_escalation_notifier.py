"""Adapter de `EscalationNotifierPort` que registra el evento en el log.

En producción se reemplaza por un adapter que publique en Kafka
(`support.escalation.created`) o llame al CRM; el núcleo no cambia.
"""

import json
import logging

from ....domain.models import Escalation
from ....logging_config import preview

logger = logging.getLogger(__name__)


class LoggingEscalationNotifier:
    def notify(self, escalation: Escalation, session_id: str, question: str) -> None:
        event = {"event": "support.escalation.created", "session_id": session_id,
                 **escalation.to_dict(), "question": preview(question)}
        logger.info("Evento de escalamiento: %s", json.dumps(event, ensure_ascii=False))
