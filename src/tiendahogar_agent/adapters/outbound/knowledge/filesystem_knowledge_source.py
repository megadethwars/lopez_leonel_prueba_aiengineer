"""Adapter de `KnowledgeSourcePort` que lee documentos `.txt` de una carpeta.

Cada archivo es un documento entregado por el cliente, copiado tal cual:
    nombre:    <doc_id>_<slug>.txt   (p. ej. doc1_politica_de_garantia.txt)
    contenido: título en la primera línea + línea en blanco + texto del documento.
"""

import logging
from pathlib import Path

from ....domain.models import Document

logger = logging.getLogger(__name__)


class FileSystemKnowledgeSource:
    def __init__(self, directory: Path | str):
        self.directory = Path(directory)

    def list_documents(self) -> list[Document]:
        documents = [self._parse(p) for p in sorted(self.directory.glob("*.txt"))]
        if not documents:
            logger.error("No hay documentos .txt en %s", self.directory)
            raise FileNotFoundError(f"No hay documentos .txt en {self.directory}")
        logger.info("Base de conocimiento cargada: %d documentos desde %s (%s)", len(documents),
                    self.directory, ", ".join(d.doc_id for d in documents))
        return documents

    @staticmethod
    def _parse(path: Path) -> Document:
        text = path.read_text(encoding="utf-8").strip()
        first_line, _, body = text.partition("\n")
        if not first_line.strip() or not body.strip():
            raise ValueError(f"{path.name}: se espera el título en la primera línea y luego el texto.")
        return Document(
            doc_id=path.stem.split("_", 1)[0],
            title=first_line.strip(),
            content=body.strip(),
        )
