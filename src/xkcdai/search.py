"""Load the index and find the comics most relevant to a piece of text.

The ``min_score`` threshold is what implements "only mention a comic if one is
genuinely relevant": queries with no good match return an empty list, so the
caller can simply stay quiet.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import numpy as np

from . import embed
from .data import load_cache
from .paths import embeddings_path, index_meta_path

logger = logging.getLogger(__name__)

# A coarse floor, not a relevance oracle. xkcd covers nearly every topic, so even
# loosely-related queries return a thematically-adjacent comic at ~0.6+. The fine
# "is this worth mentioning here?" judgment belongs to the calling LLM (see the
# score bands in the find_xkcd tool description), not to a single cutoff.
# Empirically with bge-small: strong matches ~0.75+, decent ~0.66-0.75, weak ~0.6-0.66.
DEFAULT_MIN_SCORE = 0.62


@dataclass
class Match:
    num: int
    title: str
    score: float
    url: str
    image: str
    alt: str
    explain_url: str

    def as_dict(self) -> dict:
        return asdict(self)


class Searcher:
    """Loads the index once and answers similarity queries."""

    def __init__(self) -> None:
        if not embeddings_path().exists() or not index_meta_path().exists():
            raise FileNotFoundError(
                "No index found. Build it first with:  xkcdai build"
            )
        meta = json.loads(index_meta_path().read_text(encoding="utf-8"))
        if meta.get("model") != embed.MODEL_NAME:
            raise RuntimeError(
                f"Index was built with model {meta.get('model')!r}, but the code now "
                f"uses {embed.MODEL_NAME!r}. Rebuild with:  xkcdai build --force"
            )
        self.nums: list[int] = meta["nums"]
        self.matrix: np.ndarray = np.load(embeddings_path())
        self.comics = load_cache()
        logger.info(
            "Loaded index: %d comics (model %s)", len(self.nums), meta.get("model")
        )

    def search(
        self,
        text: str,
        max_results: int = 3,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[Match]:
        text = (text or "").strip()
        if not text:
            logger.debug("empty query; returning no results")
            return []

        q = embed.embed_query(text)
        scores = self.matrix @ q  # cosine similarity (everything is normalized)

        # Take a few extra candidates, then filter by threshold.
        k = min(max_results * 4, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        results: list[Match] = []
        for idx in top:
            score = float(scores[idx])
            if score < min_score:
                break
            num = self.nums[idx]
            comic = self.comics.get(num, {})
            results.append(
                Match(
                    num=num,
                    title=comic.get("title", f"#{num}"),
                    score=round(score, 4),
                    url=f"https://xkcd.com/{num}/",
                    image=comic.get("img", ""),
                    alt=comic.get("alt", ""),
                    explain_url=f"https://www.explainxkcd.com/wiki/index.php/{num}",
                )
            )
            if len(results) >= max_results:
                break

        if results:
            top = results[0]
            logger.info(
                "search -> %d result(s); top #%d %r @%.3f",
                len(results),
                top.num,
                top.title,
                top.score,
            )
        else:
            logger.info("search -> no match (best < min_score=%.2f)", min_score)
        logger.debug("results: %s", [(m.num, m.score) for m in results])
        return results


_searcher: Searcher | None = None


def get_searcher() -> Searcher:
    """Process-wide cached Searcher (loads the model + index lazily, once)."""
    global _searcher
    if _searcher is None:
        _searcher = Searcher()
    return _searcher


def is_searcher_loaded() -> bool:
    """Whether the index is loaded (True) or not (False)."""
    return _searcher is not None
