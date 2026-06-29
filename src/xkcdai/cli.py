"""Command-line interface for building the index and testing searches.

    xkcdai enrich              # fetch transcripts/explanations from explainxkcd
    xkcdai build               # fetch new comics + (re)build the embedding index
    xkcdai build --enrich      # run enrich first, then build
    xkcdai build --force       # re-download everything and rebuild from scratch
    xkcdai search "my code finally compiled"
    xkcdai search "git merge conflict" --min-score 0.5 --max-results 5

Recommended first run:  xkcdai build  &&  xkcdai enrich  &&  xkcdai build
(or simply:  xkcdai build --enrich  on the second pass)
"""

from __future__ import annotations

import argparse
import sys

from .explain import update_explain
from .index import build
from .search import DEFAULT_MIN_SCORE, get_searcher


def _cmd_enrich(args: argparse.Namespace) -> int:
    data = update_explain(force=args.force)
    print(f"Done. explainxkcd context for {len(data)} comics.")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    if getattr(args, "enrich", False):
        update_explain()

    count = build(force_fetch=args.force)
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
    sub = parser.add_subparsers(dest="command", required=True)

    p_enrich = sub.add_parser(
        "enrich", help="fetch transcripts/explanations from explainxkcd.com"
    )
    p_enrich.add_argument(
        "--force", action="store_true", help="re-fetch all explainxkcd pages"
    )
    p_enrich.set_defaults(func=_cmd_enrich)

    p_build = sub.add_parser("build", help="fetch comics and build the embedding index")
    p_build.add_argument(
        "--force", action="store_true", help="re-download all comics before indexing"
    )
    p_build.add_argument(
        "--enrich",
        action="store_true",
        help="fetch explainxkcd context before building",
    )
    p_build.set_defaults(func=_cmd_build)

    p_search = sub.add_parser("search", help="find comics relevant to some text")
    p_search.add_argument("text", help="the text / conversation topic to match against")
    p_search.add_argument("--max-results", type=int, default=3)
    p_search.add_argument("--min-score", type=float, default=None)
    p_search.set_defaults(func=_cmd_search)

    args = parser.parse_args(argv)

    if getattr(args, "command", None) == "search" and args.min_score is None:
        args.min_score = DEFAULT_MIN_SCORE

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
