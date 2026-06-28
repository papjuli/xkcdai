"""Shared filesystem locations for cached data and the built index.

Override the location with the ``XKCDAI_DATA_DIR`` environment variable. This
matters for the MCP server, which is launched by a host (Claude Desktop / Claude
Code) with an unpredictable working directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Directory holding the comic cache and the embedding index."""
    env = os.environ.get("XKCDAI_DATA_DIR")
    if env:
        base = Path(env).expanduser()
    else:
        # Default to a `data/` folder next to the repo root (src/xkcdai/ -> ../../data).
        base = Path(__file__).resolve().parents[2] / "data"
    base.mkdir(parents=True, exist_ok=True)
    return base


def comics_path() -> Path:
    return data_dir() / "comics.json"


def embeddings_path() -> Path:
    return data_dir() / "embeddings.npy"


def index_meta_path() -> Path:
    return data_dir() / "index.json"
