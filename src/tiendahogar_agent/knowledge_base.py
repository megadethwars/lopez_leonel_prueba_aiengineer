"""Carga la base de conocimiento desde `data/knowledge_base/*.txt`.

Cada archivo es un documento entregado por el cliente, copiado tal cual:
    nombre:    <doc_id>_<slug>.txt   (p. ej. doc1_politica_de_garantia.txt)
    contenido: título en la primera línea + línea en blanco + texto del documento.

Para agregar conocimiento basta con añadir un archivo; el índice se construye al
arrancar. Cada documento es corto (1 párrafo), por lo que se indexa como un
único chunk; ver SUBMISSION.md.
"""

import os
from dataclasses import dataclass
from pathlib import Path

HUMAN_SUPPORT_EMAIL = "soporte@tiendahogar.example"

DEFAULT_KB_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    content: str


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


def load_documents(kb_dir: Path | str | None = None) -> list[Document]:
    directory = Path(kb_dir or os.getenv("KNOWLEDGE_BASE_DIR") or DEFAULT_KB_DIR)
    documents = [_parse(p) for p in sorted(directory.glob("*.txt"))]
    if not documents:
        raise FileNotFoundError(f"No hay documentos .txt en {directory}")
    return documents


DOCUMENTS: list[Document] = load_documents()
