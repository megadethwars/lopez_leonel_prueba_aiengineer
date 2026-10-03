"""Adapter de `SkillRepositoryPort` que lee `<carpeta>/<nombre>/SKILL.md`.

Una skill define CÓMO conversa el agente (tono, empatía, formato) y puede editarla
alguien de negocio o CX sin tocar código.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _strip_frontmatter(text: str) -> str:
    """Quita el bloque YAML inicial (--- ... ---) si existe."""
    if text.startswith("---"):
        _, _, rest = text[3:].partition("\n---")
        return rest.lstrip("-").strip()
    return text.strip()


class FileSystemSkillRepository:
    def __init__(self, directory: Path | str):
        self.directory = Path(directory)

    def get(self, name: str) -> str:
        path = self.directory / name / "SKILL.md"
        if not path.is_file():
            logger.error("No se encontró la skill %s en %s", name, path)
            raise FileNotFoundError(path)
        body = _strip_frontmatter(path.read_text(encoding="utf-8"))
        logger.info("Skill '%s' cargada desde %s (%d caracteres)", name, path, len(body))
        return body
