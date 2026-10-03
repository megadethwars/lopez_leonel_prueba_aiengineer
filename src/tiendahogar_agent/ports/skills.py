"""Port del repositorio de skills (instrucciones de conversación).

Adapters posibles: archivos SKILL.md (actual), un servicio de gestión de prompts,
una tabla versionada, etc.
"""

from typing import Protocol


class SkillRepositoryPort(Protocol):
    def get(self, name: str) -> str:
        """Instrucciones de la skill, sin metadatos."""
        ...
