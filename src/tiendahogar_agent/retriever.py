"""Recuperación híbrida en memoria: embeddings densos + coincidencia léxica (IDF).

score = alpha * coseno(embedding) + (1 - alpha) * léxico

- El componente denso captura paráfrasis ("¿me devuelven mi dinero?" → Reembolsos).
- El léxico rescata consultas cortas o con términos exactos del dominio
  ("¿Garantía de una licuadora?"), donde MiniLM da similitudes bajas.
Solo se usan documentos con score >= umbral (máximo top-k).
"""

import logging
import math
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np

from .config import settings
from .guardrails import normalize
from .knowledge_base import DOCUMENTS, Document
from .logging_config import preview

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedEmbedder:
    """Embeddings multilingües locales vía ONNX (sin GPU, sin API key)."""

    def __init__(self, model_name: str | None = None):
        from fastembed import TextEmbedding

        name = model_name or settings.embedding_model
        started = time.perf_counter()
        try:
            self._model = TextEmbedding(model_name=name)
        except Exception:
            logger.exception("No se pudo cargar el modelo de embeddings %s", name)
            raise
        logger.info("Modelo de embeddings %s cargado en %.1f s", name, time.perf_counter() - started)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.embed(texts)), dtype=np.float32)


_STOPWORDS = set(
    "de la el los las un una unos unas que y o a en por para con mi mis tu su sus es son se lo "
    "le me te al del como cual cuanto cuando donde puedo tiene tienen hay hace esta este esto "
    "pero mas muy sin sobre".split()
)


def _terms(text: str) -> set[str]:
    """Términos normalizados con stemming por prefijo (licuadora/licuadoras → 'licua')."""
    words = re.findall(r"[a-z0-9]+", normalize(text))
    return {w[:5] for w in words if len(w) > 2 and w not in _STOPWORDS}


@dataclass(frozen=True)
class RetrievedDoc:
    document: Document
    score: float


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        documents: list[Document] = DOCUMENTS,
        top_k: int | None = None,
        threshold: float | None = None,
        alpha: float | None = None,
    ):
        self.embedder = embedder
        self.documents = documents
        self.top_k = top_k if top_k is not None else settings.retrieval_top_k
        self.threshold = threshold if threshold is not None else settings.retrieval_threshold
        self.alpha = alpha if alpha is not None else settings.retrieval_alpha
        texts = [f"{d.title}. {d.content}" for d in documents]
        # Se indexa título + contenido: el título aporta señal semántica extra.
        self._matrix = self._normalize(embedder.embed(texts))
        self._doc_terms = [_terms(t) for t in texts]
        df: dict[str, int] = {}
        for terms in self._doc_terms:
            for term in terms:
                df[term] = df.get(term, 0) + 1
        self._df = df
        logger.info(
            "Índice RAG construido: %d documentos, %d dimensiones, top_k=%d, umbral=%.2f, alpha=%.2f",
            len(documents), self._matrix.shape[1], self.top_k, self.threshold, self.alpha,
        )

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)

    def _idf(self, term: str) -> float:
        # Términos fuera del corpus pesan más: diluyen el score de consultas fuera de dominio.
        return math.log(1 + len(self.documents) / self._df.get(term, 0.5))

    def _lexical(self, query: str) -> np.ndarray:
        q_terms = _terms(query)
        total = sum(self._idf(t) for t in q_terms)
        if not total:
            return np.zeros(len(self.documents), dtype=np.float32)
        return np.array(
            [sum(self._idf(t) for t in q_terms & doc) / total for doc in self._doc_terms],
            dtype=np.float32,
        )

    def scores(self, query: str) -> list[RetrievedDoc]:
        """Todos los documentos con su score híbrido, de mayor a menor."""
        q = self._normalize(self.embedder.embed([query]))[0]
        hybrid = self.alpha * (self._matrix @ q) + (1 - self.alpha) * self._lexical(query)
        order = np.argsort(-hybrid)
        ranked = [RetrievedDoc(self.documents[i], float(hybrid[i])) for i in order]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Scores para %r: %s", preview(query),
                         ", ".join(f"{r.document.doc_id}={r.score:.3f}" for r in ranked))
        return ranked

    def retrieve(self, query: str) -> list[RetrievedDoc]:
        """Top-k documentos cuyo score supera el umbral (lista vacía si ninguno)."""
        return [r for r in self.scores(query)[: self.top_k] if r.score >= self.threshold]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(FastEmbedEmbedder())
