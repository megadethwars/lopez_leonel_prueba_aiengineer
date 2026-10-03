"""Proveedores de embeddings para el retriever híbrido.

`Embedder` es un detalle interno del adapter de recuperación (no un port del
núcleo): permite cambiar fastembed por Azure OpenAI, Databricks Model Serving,
etc., sin tocar el retriever.
"""

import logging
import time
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedEmbedder:
    """Embeddings multilingües locales vía ONNX (sin GPU, sin API key)."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        started = time.perf_counter()
        try:
            self._model = TextEmbedding(model_name=model_name)
        except Exception:
            logger.exception("No se pudo cargar el modelo de embeddings %s", model_name)
            raise
        logger.info("Modelo de embeddings %s cargado en %.1f s", model_name,
                    time.perf_counter() - started)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.embed(texts)), dtype=np.float32)
