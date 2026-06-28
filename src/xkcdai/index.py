"""Build the embedding index from the cached comics + explainxkcd context.

Output (in the data dir):
  - embeddings.npy : float32 (N, dim), L2-normalized, row i == nums[i]
  - index.json     : {model, dim, nums: [...]} mapping rows back to comic numbers

With only a few thousand comics, a plain numpy matrix + a dot product is faster
and simpler than any vector database.
"""

from __future__ import annotations

import json

import numpy as np

from . import embed
from .data import document_text, update_cache
from .explain import load_explain
from .paths import embeddings_path, index_meta_path


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


def build(force_fetch: bool = False) -> int:
    """Fetch (if needed), embed all comics, and write the index. Returns count."""
    comics = update_cache(force=force_fetch)
    explain = load_explain()
    if explain:
        print(f"Using explainxkcd context for {len(explain)} comics.")
    else:
        print("No explainxkcd context found — run `xkcdai enrich` for better matches.")

    nums: list[int] = []
    texts: list[str] = []
    for num in sorted(comics):
        text = build_document(comics[num], explain.get(num))
        if text:
            nums.append(num)
            texts.append(text)

    if not texts:
        raise RuntimeError("No comics with text to index. Did the fetch fail?")

    print(f"Embedding {len(texts)} comics with {embed.MODEL_NAME} ...")
    matrix = embed.embed_documents(texts)

    np.save(embeddings_path(), matrix)
    index_meta_path().write_text(
        json.dumps({"model": embed.MODEL_NAME, "dim": embed.EMBED_DIM, "nums": nums}),
        encoding="utf-8",
    )
    print(f"Wrote {matrix.shape[0]} embeddings -> {embeddings_path()}")
    return matrix.shape[0]
