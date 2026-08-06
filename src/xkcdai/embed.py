"""Thin wrapper around the embedding model.

We use fastembed (ONNX runtime) instead of sentence-transformers so there's no
PyTorch dependency — much lighter to install and run as an MCP server. Models are
downloaded once and cached locally, then run fully offline.

bge models are asymmetric: documents and queries are embedded differently
(queries get an instruction prefix). fastembed handles this via ``passage_embed``
and ``query_embed``. Swapping the model is a one-line change to ``MODEL_NAME``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384-dim, good quality/size tradeoff
EMBED_DIM = 384

_model = None


def _get_model():
    global _model
    if _model is None:
        import os

        from fastembed import TextEmbedding  # imported lazily; heavy import

        # FASTEMBED_CACHE_DIR lets a container bake the model into a stable path at
        # build time (see Dockerfile), avoiding a download on every cold start.
        cache_dir = os.environ.get("FASTEMBED_CACHE_DIR") or None

        # XKCDAI_ORT_THREADS opts into a small-instance profile (the Dockerfile
        # sets it): cap ONNX Runtime's thread pool and drop its allocation arena.
        # Unset by default — `xkcdai build` wants every core it can get.
        opts: dict[str, Any] = {}
        threads = os.environ.get("XKCDAI_ORT_THREADS")
        if threads:
            try:
                threads = int(threads)
            except ValueError:
                logger.warning(
                    "XKCDAI_ORT_THREADS=%r is not an integer; ignoring", threads
                )
                threads = None
        if threads:
            opts["threads"] = threads
            opts["enable_cpu_mem_arena"] = False

        logger.debug(
            "loading embedding model %s (cache_dir=%s, opts=%s)",
            MODEL_NAME,
            cache_dir,
            opts or "onnxruntime defaults",
        )
        _model = TextEmbedding(model_name=MODEL_NAME, cache_dir=cache_dir, **opts)
    return _model


def _normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_documents(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed comic documents. Returns an L2-normalized (N, EMBED_DIM) float32 array."""
    model = _get_model()
    vecs = list(model.passage_embed(texts, batch_size=batch_size))
    arr = np.asarray(vecs, dtype=np.float32)
    return _normalize(arr)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query. Returns an L2-normalized (EMBED_DIM,) float32 vector."""
    model = _get_model()
    vec = next(iter(model.query_embed([text])))
    arr = np.asarray(vec, dtype=np.float32)[None, :]
    return _normalize(arr)[0]
