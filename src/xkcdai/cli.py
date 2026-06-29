"""Command-line interface for building the index and testing searches.

    xkcdai build               # fetch comics + explainxkcd context, then (re)embed
    xkcdai build --no-enrich   # skip explainxkcd fetch (faster/offline; weaker matches)
    xkcdai build --force       # re-download all comics and rebuild from scratch
    xkcdai enrich              # only fetch transcripts/explanations from explainxkcd
    xkcdai search "my code finally compiled"
    xkcdai search "git merge conflict" --min-score 0.5 --max-results 5

`xkcdai build` is all you need for first run and for picking up new comics — it
fetches comics, then their explainxkcd context, then embeds (all incremental).
"""

from __future__ import annotations

import argparse
import logging
import sys

from .explain import update_explain
from .index import build
from .search import DEFAULT_MIN_SCORE, get_searcher


def _configure_logging(verbose: bool) -> None:
    """Send library logs to stderr; ``-v`` enables DEBUG-level detail."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(levelname)s %(name)s: %(message)s" if verbose else "%(message)s"
    logging.basicConfig(level=level, format=fmt)
    # httpx/httpcore log every request at INFO — far too chatty for a bulk fetch.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _cmd_enrich(args: argparse.Namespace) -> int:
    data = update_explain(force=args.force)
    print(f"Done. explainxkcd context for {len(data)} comics.")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    count = build(force_fetch=args.force, enrich=not args.no_enrich)
    print(f"Done. Indexed {count} comics.")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    matches = get_searcher().search(
        args.text, max_results=args.max_results, min_score=args.min_score
    )
    if not matches:
        print("(no relevant comic — nothing above the threshold)")
        return 0
    for m in matches:
        print(f"[{m.score:.3f}] #{m.num}  {m.title}")
        print(f"        {m.url}")
        print(f"        alt: {m.alt}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xkcdai", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v", "--verbose", action="store_true", help="show detailed debug logging"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_enrich = sub.add_parser(
        "enrich",
        parents=[common],
        help="fetch transcripts/explanations from explainxkcd.com",
    )
    p_enrich.add_argument(
        "--force", action="store_true", help="re-fetch all explainxkcd pages"
    )
    p_enrich.set_defaults(func=_cmd_enrich)

    p_build = sub.add_parser(
        "build",
        parents=[common],
        help="fetch comics + explainxkcd context and build the index",
    )
    p_build.add_argument(
        "--force", action="store_true", help="re-download all comics before indexing"
    )
    p_build.add_argument(
        "--no-enrich",
        action="store_true",
        help="skip the explainxkcd fetch (faster/offline; weaker matches)",
    )
    p_build.set_defaults(func=_cmd_build)

    p_search = sub.add_parser(
        "search", parents=[common], help="find comics relevant to some text"
    )
    p_search.add_argument("text", help="the text / conversation topic to match against")
    p_search.add_argument("--max-results", type=int, default=3)
    p_search.add_argument("--min-score", type=float, default=None)
    p_search.set_defaults(func=_cmd_search)

    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))

    if getattr(args, "command", None) == "search" and args.min_score is None:
        args.min_score = DEFAULT_MIN_SCORE

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
