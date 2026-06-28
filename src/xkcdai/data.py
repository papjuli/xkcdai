"""Fetch and cache xkcd comic metadata from the official JSON API.

xkcd exposes one JSON document per comic:

    https://xkcd.com/info.0.json        -> the latest comic (gives us the max num)
    https://xkcd.com/<num>/info.0.json  -> a specific comic

Each document includes ``num``, ``title``, ``alt`` (the mouseover text),
``transcript`` (present for most older comics), ``img``, and the date. We cache
everything to ``comics.json`` and only fetch numbers we don't already have, so
re-running ``build`` after a few new comics is cheap.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from tqdm import tqdm

from .paths import comics_path

LATEST_URL = "https://xkcd.com/info.0.json"
COMIC_URL = "https://xkcd.com/{num}/info.0.json"

# Comic #404 deliberately 404s — it's the joke. Never try to fetch it.
SKIP = {404}

# Fields we keep. (xkcd returns a few more, e.g. news/link, that we don't need.)
KEEP_FIELDS = (
    "num",
    "title",
    "safe_title",
    "alt",
    "transcript",
    "img",
    "year",
    "month",
    "day",
)


def _slim(doc: dict) -> dict:
    return {k: doc.get(k, "") for k in KEEP_FIELDS}


def load_cache() -> dict[int, dict]:
    """Load cached comics as {num: comic}. Empty dict if nothing cached yet."""
    path = comics_path()
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def save_cache(comics: dict[int, dict]) -> None:
    path = comics_path()
    ordered = {str(n): comics[n] for n in sorted(comics)}
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=0), encoding="utf-8")


def get_latest_num(client: httpx.Client) -> int:
    resp = client.get(LATEST_URL, timeout=30)
    resp.raise_for_status()
    return int(resp.json()["num"])


def _fetch_one(client: httpx.Client, num: int) -> dict | None:
    try:
        resp = client.get(COMIC_URL.format(num=num), timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return _slim(resp.json())
    except (httpx.HTTPError, json.JSONDecodeError):
        return None


def update_cache(workers: int = 16, force: bool = False) -> dict[int, dict]:
    """Fetch every comic not already cached and return the full {num: comic} map.

    Set ``force=True`` to re-download everything from scratch.
    """
    cached = {} if force else load_cache()
    headers = {
        "User-Agent": "xkcdai/0.1 (+https://github.com/; semantic xkcd suggester)"
    }

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        latest = get_latest_num(client)
        missing: list[int] = [
            n for n in range(1, latest + 1) if n not in SKIP and n not in cached
        ]

        if not missing:
            print(f"Cache is up to date ({len(cached)} comics, latest #{latest}).")
            return cached

        print(f"Fetching {len(missing)} new comic(s) (latest is #{latest})...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch_one, client, n): n for n in missing}
            for fut in tqdm(as_completed(futures), total=len(futures), unit="comic"):
                doc = fut.result()
                if doc is not None:
                    cached[int(doc["num"])] = doc

    save_cache(cached)
    print(f"Cached {len(cached)} comics -> {comics_path()}")
    return cached


def document_text(comic: dict) -> str:
    """The text we embed for a comic: title + mouseover alt + transcript.

    The transcript (when present) is what makes semantic matching work well — it
    describes what's actually happening in the panels.
    """
    parts: list[str] = []
    title = comic.get("title") or comic.get("safe_title") or ""
    if title:
        parts.append(title)
    alt = comic.get("alt") or ""
    if alt:
        parts.append(alt)
    transcript = comic.get("transcript") or ""
    if transcript:
        parts.append(transcript)
    return "\n".join(parts).strip()


def comics_in_order(comics: dict[int, dict]) -> list[dict]:
    """Comics that have embeddable text, sorted by number."""
    out: list[dict] = []
    for num in sorted(comics):
        if document_text(comics[num]):
            out.append(comics[num])
    return out
