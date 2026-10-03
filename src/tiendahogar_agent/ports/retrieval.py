"""Port de recuperación (RAG).

Adapters posibles: índice híbrido en memoria (actual), Mosaic AI Vector Search
(Databricks), Azure AI Search, pgvector, etc.
"""

from typing import Protocol

from ..domain.models import RetrievedDocument


class RetrieverPort(Protocol):
    def retrieve(self, query: str) -> list[RetrievedDocument]:
        """Documentos relevantes para la consulta, de mayor a menor score.

        Debe devolver solo los que superan el umbral de relevancia (lista vacía si ninguno).
        """
        ...
