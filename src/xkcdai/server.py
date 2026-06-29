"""MCP server exposing a single tool that finds a relevant xkcd comic.

Two transports, selected by the ``XKCDAI_TRANSPORT`` environment variable:

  * ``stdio`` (default) — for local hosts like Claude Code / Claude Desktop, which
    launch this as a subprocess.  ``python -m xkcdai.server``
  * ``streamable-http`` — for hosting publicly so it can be added as a Claude
    *custom connector* (works on the web + mobile apps). Binds ``0.0.0.0:$PORT``
    and serves MCP at ``/mcp``. See the Dockerfile and the README.
"""

from __future__ import annotations

import logging
import os

from mcp.server.fastmcp import FastMCP

from .search import DEFAULT_MIN_SCORE, get_searcher


def _configure_logging() -> None:
    """Surface xkcdai.* logs (incl. each search) on stderr.

    The CLI sets up logging itself; the server is launched by a host (Claude
    Desktop/Code, or uvicorn when hosted), so we configure the package logger
    here. stderr is mandatory: in stdio mode stdout carries the MCP protocol.
    Set XKCDAI_VERBOSE=1 for DEBUG detail. We own the ``xkcdai`` logger (and don't
    propagate) so uvicorn's logging config can't suppress or duplicate it.
    """
    level = logging.DEBUG if os.environ.get("XKCDAI_VERBOSE") else logging.INFO
    pkg = logging.getLogger("xkcdai")
    pkg.setLevel(level)
    if not pkg.handlers:
        handler = logging.StreamHandler()  # defaults to stderr
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        pkg.addHandler(handler)
    pkg.propagate = False


# host/port only matter for the HTTP transport; harmless for stdio.
mcp = FastMCP(
    "xkcdai",
    host=os.environ.get("HOST", "0.0.0.0"),
    port=int(os.environ.get("PORT", "8000")),
)


@mcp.tool()
def find_xkcd(
    context: str,
    max_results: int = 3,
    min_score: float = DEFAULT_MIN_SCORE,
) -> dict:
    """Find xkcd comics semantically relevant to the current conversation.

    Call this whenever an xkcd comic might enrich the conversation — when the
    discussion lands on a topic xkcd is famous for skewering (programming, science,
    statistics, relationships, the absurdity of standards, etc.).

    Pass a concise description of the current topic or theme as `context` (a phrase
    or sentence works better than a whole transcript), e.g. "spending hours
    automating a task that was faster to do by hand" or "code finally compiling".

    IMPORTANT — deciding whether to mention one. xkcd has a comic for almost every
    topic, so this tool will nearly always return something. A result being
    returned does NOT mean you should bring it up. Use the `score` as a signal and
    apply your own judgment about conversational fit:
        score >= 0.75  strong match — usually worth mentioning if it fits the moment
        0.66 - 0.75    plausible — mention only if it genuinely lands
        < 0.66         weak/tangential — almost always better to stay silent
    Only one comic, at most, per topic — and only when it actually adds something.

    When you do share one, cite it by number and title with its `url`, and quote
    the `alt` (mouseover) text — it's half the joke.

    Returns a dict with a `results` list (num, title, score, url, image, alt,
    explain_url) and a `count`. An empty list means nothing cleared the floor.
    """
    try:
        searcher = get_searcher()
    except FileNotFoundError as e:
        return {"error": str(e), "results": [], "count": 0}

    matches = searcher.search(context, max_results=max_results, min_score=min_score)
    return {
        "query": context,
        "count": len(matches),
        "results": [m.as_dict() for m in matches],
    }


def main() -> None:
    _configure_logging()
    transport = os.environ.get("XKCDAI_TRANSPORT", "stdio").lower().replace("_", "-")
    if transport in ("http", "streamable-http"):
        mcp.run(transport="streamable-http")
    elif transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run()  # stdio


if __name__ == "__main__":
    main()
