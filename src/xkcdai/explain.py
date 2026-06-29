"""Pull extra context for each comic from explainxkcd.com.

explainxkcd.com is a community MediaWiki that, for nearly every comic, has both a
**Transcript** (the literal panel text) and an **Explanation** (what the comic is
actually about). Both are excellent material for semantic search. We fetch the raw
wikitext via the MediaWiki API, extract those two sections, lightly strip markup,
and cache to ``explain.json``.

Be a good citizen: this hits a volunteer-run wiki, so we use modest concurrency,
a descriptive User-Agent, and cache aggressively (only fetch numbers we're missing).
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from tqdm import tqdm

from .data import load_cache
from .paths import data_dir

logger = logging.getLogger(__name__)

API_URL = "https://www.explainxkcd.com/wiki/api.php"

# Cap the explanation we keep — the first part carries the gist; the rest adds
# length (and noise) without much retrieval benefit.
EXPLANATION_MAX_CHARS = 1200


def explain_path():
    return data_dir() / "explain.json"


def load_explain() -> dict[int, dict]:
    path = explain_path()
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def save_explain(data: dict[int, dict]) -> None:
    ordered = {str(n): data[n] for n in sorted(data)}
    explain_path().write_text(
        json.dumps(ordered, ensure_ascii=False, indent=0), encoding="utf-8"
    )


# --- wikitext parsing -------------------------------------------------------

_LEVEL2 = re.compile(r"^==[^=].*?==\s*$", re.MULTILINE)


def _section(wikitext: str, name: str) -> str:
    """Return the body of a level-2 ``== name ==`` section, or ''."""
    header = re.compile(rf"^==\s*{re.escape(name)}\s*==\s*$", re.MULTILINE)
    m = header.search(wikitext)
    if not m:
        return ""
    start = m.end()
    nxt = _LEVEL2.search(wikitext, start)
    end = nxt.start() if nxt else len(wikitext)
    return wikitext[start:end].strip()


def _strip_markup(text: str) -> str:
    """Reduce wikitext to plain-ish text good enough for embedding."""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)  # comments
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)  # simple templates
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)  # nested (2nd pass)
    text = re.sub(r"\[\[File:[^\]]*\]\]", " ", text, flags=re.I)  # images
    text = re.sub(r"\[\[Category:[^\]]*\]\]", " ", text, flags=re.I)  # categories
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)  # [[link|label]]
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)  # [[link]]
    text = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", text)  # [url label]
    text = re.sub(r"\[https?://\S+\]", " ", text)  # [url]
    text = re.sub(r"</?[^>]+>", " ", text)  # html tags
    text = re.sub(r"'''?", "", text)  # bold/italic
    text = re.sub(r"^[*#:;]+", "", text, flags=re.MULTILINE)  # list/indent markers
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fetch_one(client: httpx.Client, num: int) -> dict | None:
    params = {
        "action": "parse",
        "page": str(num),
        "prop": "wikitext",
        "redirects": "1",
        "format": "json",
        "formatversion": "2",
    }
    try:
        resp = client.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.warning("explainxkcd fetch failed for #%d: %s", num, e)
        return None
    parse = data.get("parse")
    if not parse:
        logger.debug("explainxkcd returned no wikitext for #%d", num)
        return None
    wt = parse.get("wikitext", "")
    transcript = _strip_markup(_section(wt, "Transcript"))
    explanation = _strip_markup(_section(wt, "Explanation"))
    if len(explanation) > EXPLANATION_MAX_CHARS:
        logger.debug(
            "explanation for #%d is %d chars; truncating to %d",
            num,
            len(explanation),
            EXPLANATION_MAX_CHARS,
        )
        explanation = explanation[:EXPLANATION_MAX_CHARS].rsplit(" ", 1)[0]
    if not transcript and not explanation:
        logger.debug("no transcript or explanation found for #%d", num)
        return None
    return {"transcript": transcript, "explanation": explanation}


def update_explain(workers: int = 6, force: bool = False) -> dict[int, dict]:
    """Fetch explainxkcd context for every comic we don't already have it for."""
    cached = {} if force else load_explain()
    comics = load_cache()
    if not comics:
        raise RuntimeError("No comics cached yet. Run `xkcdai build` first.")

    missing = [n for n in sorted(comics) if n not in cached]
    if not missing:
        logger.info("explainxkcd cache is up to date (%d entries).", len(cached))
        return cached

    headers = {
        "User-Agent": "xkcdai/0.1 (semantic xkcd suggester; contact: papjuli@gmail.com)"
    }
    logger.info("Fetching explainxkcd context for %d comic(s)...", len(missing))
    skipped = 0
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch_one, client, n): n for n in missing}
            for fut in tqdm(as_completed(futures), total=len(futures), unit="page"):
                num = futures[fut]
                doc = fut.result()
                if doc is not None:
                    cached[num] = doc
                else:
                    skipped += 1

    if skipped:
        logger.info(
            "%d comic(s) had no usable explainxkcd content (use -v for details).",
            skipped,
        )
    save_explain(cached)
    logger.info(
        "Cached explainxkcd context for %d comics -> %s", len(cached), explain_path()
    )
    return cached
