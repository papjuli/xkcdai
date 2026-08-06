"""xkcdai: find the xkcd comic relevant to a conversation, via semantic search."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth is pyproject.toml; read it back from the installed
    # metadata rather than restating it here, where the two would drift apart.
    __version__ = version("xkcdai")
except PackageNotFoundError:
    # Imported from a source tree that was never installed (e.g. PYTHONPATH=src).
    __version__ = "0.0.0+unknown"
