"""Port de la fuente de la base de conocimiento.

Adapters posibles: archivos .txt (actual), tablas Delta en Unity Catalog, SharePoint,
un CMS, un bucket S3/Blob, etc.
"""

from typing import Protocol

from ..domain.models import Document


class KnowledgeSourcePort(Protocol):
    def list_documents(self) -> list[Document]:
        """Todos los documentos vigentes de la base de conocimiento."""
        ...
