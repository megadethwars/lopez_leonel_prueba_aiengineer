"""Utilidades de texto del dominio."""

import unicodedata


def normalize(text: str) -> str:
    """Minúsculas y sin tildes, para que las reglas no dependan de la ortografía."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()
