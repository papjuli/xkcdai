"""Build the embedding index from the cached comics + explainxkcd context.

Output (in the data dir):
  - embeddings.npy : float32 (N, dim), L2-normalized, row i == nums[i]
  - index.json     : {model, dim, nums: [...]} mapping rows back to comic numbers

With only a few thousand comics, a plain numpy matrix + a dot product is faster
and simpler than any vector database.
"""

from __future__ import annotations

import json
import logging

import numpy as np

from . import embed
from .data import document_text, update_cache
from .explain import load_explain, update_explain
from .paths import embeddings_path, index_meta_path

logger = logging.getLogger(__name__)


def build_document(comic: dict, explain_entry: dict | None) -> str:
    """The text we embed for one comic.

    Combines the official metadata (title, mouseover alt, transcript when present)
    with explainxkcd's community transcript and explanation. The explainxkcd data
    is what makes newer, transcript-less comics matchable — and the explanation
    describes what the comic is *about*, which sharpens semantic matching across
    the board.
    """
    parts: list[str] = [document_text(comic)]  # title + alt + official transcript

    if explain_entry:
        # Only add the community transcript if the official one was missing, to
        # avoid embedding the same panel text twice.
        if not (comic.get("transcript") or "").strip():
            tr = explain_entry.get("transcript", "")
            if tr:
                parts.append(tr)
        expl = explain_entry.get("explanation", "")
        if expl:
            parts.append(expl)

    return "\n".join(p for p in parts if p).strip()


def build(force_fetch: bool = False, enrich: bool = True) -> int:
    """Fetch (if needed), embed all comics, and write the index. Returns count.

    With ``enrich=True`` (the default) explainxkcd context is fetched after the
    comics, so brand-new comics get their transcript/explanation before embedding.
    Pass ``enrich=False`` to skip the explainxkcd network calls and embed with only
    whatever context is already cached (faster/offline, but weaker matches).
    """
    comics = update_cache(force=force_fetch)
    explain = update_explain() if enrich else load_explain()
    if not explain:
        logger.warning(
            "No explainxkcd context — matches will be weaker. (Drop --no-enrich.)"
        )

    nums: list[int] = []
    texts: list[str] = []
    skipped = 0
    for num in sorted(comics):
        text = build_document(comics[num], explain.get(num))
        if text:
            nums.append(num)
            texts.append(text)
        else:
            skipped += 1
            logger.debug("comic #%d has no embeddable text; skipping", num)

    if not texts:
        raise RuntimeError("No comics with text to index. Did the fetch fail?")

    logger.debug(
        "indexing %d comics (%d with explainxkcd context, %d skipped as empty)",
        len(texts),
        sum(1 for n in nums if n in explain),
        skipped,
    )
    logger.info("Embedding %d comics with %s ...", len(texts), embed.MODEL_NAME)
    matrix = embed.embed_documents(texts)

    np.save(embeddings_path(), matrix)
    index_meta_path().write_text(
        json.dumps({"model": embed.MODEL_NAME, "dim": embed.EMBED_DIM, "nums": nums}),
        encoding="utf-8",
    )
    logger.info("Wrote %d embeddings -> %s", matrix.shape[0], embeddings_path())
    return matrix.shape[0]
