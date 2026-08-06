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
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import __version__
from .search import DEFAULT_MIN_SCORE, get_searcher, is_searcher_loaded

logger = logging.getLogger("xkcdai.server")


def _configure_logging() -> None:
    """Surface xkcdai.* logs (incl. each search) on stderr, one line per record.

    The CLI sets up logging itself; the server is launched by a host (Claude
    Desktop/Code, or uvicorn when hosted), so we configure the package logger
    here. stderr is mandatory: in stdio mode stdout carries the MCP protocol.
    Set XKCDAI_VERBOSE=1 for DEBUG detail. We own the ``xkcdai`` logger (and don't
    propagate) so uvicorn's logging config can't suppress or duplicate it.

    We also take the root logger back from FastMCP and reset the handler to a plain one, which keeps one record on one line.
    """
    level = logging.DEBUG if os.environ.get("XKCDAI_VERBOSE") else logging.INFO

    root_handler = logging.StreamHandler()  # defaults to stderr
    root_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    # force=True drops the handlers FastMCP already installed, RichHandler included.
    logging.basicConfig(level=logging.INFO, handlers=[root_handler], force=True)

    pkg = logging.getLogger("xkcdai")
    pkg.setLevel(level)
    if not pkg.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        pkg.addHandler(handler)
    pkg.propagate = False


# host/port only matter for the HTTP transport; harmless for stdio.
# stateless_http/json_response likewise: find_xkcd keeps no per-client state, so
# sessions would only pile up (the SDK drops them on an explicit DELETE that real
# clients rarely send) until the instance OOMs, and a session id bound to a dead
# process 404s every client after a restart.
mcp = FastMCP(
    "xkcdai",
    host=os.environ.get("HOST", "0.0.0.0"),
    port=int(os.environ.get("PORT", "8000")),
    stateless_http=True,
    json_response=True,
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


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(_request: Request) -> JSONResponse:
    """Liveness probe that also reports the footprint.

    Tells you whether the process is up, which build is
    answering, whether the index actually loaded,
    and what it currently weighs.
    ``version`` is xkcdai's (note that the MCP handshake's
    ``serverInfo.version`` reports the SDK's version).

    ``request`` is unused but required: custom_route hands every handler the
    Starlette request.
    """
    return JSONResponse(
        {
            "status": "ok",
            "version": __version__,
            "index_loaded": is_searcher_loaded(),
            "rss_mb": _rss_mb(),
        }
    )


def _rss_mb() -> float | None:
    """Resident set size in MB, or None where we can't cheaply tell (non-Linux)."""
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def _warm_up() -> None:
    """Load the model and index now rather than during the first tool call.

    Everything the server needs is baked into the image, so this is fast — but it
    also pins down the steady-state footprint at boot, where it's visible in the
    logs, instead of letting it appear mid-request on a memory-tight instance.
    """
    try:
        get_searcher().search("warm up the embedding model")
    except (FileNotFoundError, RuntimeError) as e:
        logger.warning("warm-up skipped: %s", e)
        return
    rss = _rss_mb()
    logger.info("Warm-up complete%s", f" (RSS {rss:.0f} MB)" if rss is not None else "")


def main() -> None:
    _configure_logging()
    transport = os.environ.get("XKCDAI_TRANSPORT", "stdio").lower().replace("_", "-")
    if transport in ("http", "streamable-http"):
        _warm_up()
        mcp.run(transport="streamable-http")
    elif transport == "sse":
        _warm_up()
        mcp.run(transport="sse")
    else:
        mcp.run()  # stdio: stay lazy so the host's launch isn't blocked


if __name__ == "__main__":
    main()
