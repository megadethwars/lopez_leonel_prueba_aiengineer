"""Agente de soporte al cliente de TiendaHogar (RAG + tools + guardrails)."""

try:  # Usa el almacén de certificados del SO (redes con proxy/antivirus que inspeccionan TLS).
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover
    pass
