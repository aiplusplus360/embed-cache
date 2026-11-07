"""CLI:

    embedcache --db cache.db embed --model demo "some text"
    embedcache --db cache.db stats
    embedcache --db cache.db purge            # everything
    embedcache --db cache.db purge --model x  # one model

``--model`` lives on each subcommand rather than at the top level, because its
meaning differs: ``embed`` needs one, while ``stats`` and ``purge`` treat
"omitted" as "all models". A single inherited flag made `purge` silently delete
one model and report success.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from .cache import EmbeddingCache
from .demo import hashing_embedder
from .store import SqliteStore, StoreError

EXIT_OK = 0
EXIT_ERROR = 2


def cmd_embed(args: argparse.Namespace) -> int:
    with SqliteStore(args.db) as store:
        cache = EmbeddingCache(store, args.model, hashing_embedder(args.dim))
        vectors = cache.embed(args.text)
        print(json.dumps({
            "model": args.model,
            "vectors": vectors,
            "stats": cache.stats.as_dict(),
        }, indent=2))
    return EXIT_OK


def cmd_stats(args: argparse.Namespace) -> int:
    with SqliteStore(args.db) as store:
        print(json.dumps({
            "db": args.db,
            "model": args.model,
            "entries": store.count(args.model),
            "models": store.models(),
            "cached_characters": store.characters(args.model),
        }, indent=2))
    return EXIT_OK


def cmd_purge(args: argparse.Namespace) -> int:
    with SqliteStore(args.db) as store:
        removed = store.purge(args.model)
        print(json.dumps({"removed": removed, "model": args.model}, indent=2))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embedcache",
        description="Content-addressed embedding cache. No server, no Redis.",
    )
    parser.add_argument("--db", default="embeddings.db", help="SQLite file")
    sub = parser.add_subparsers(dest="command", required=True)

    embed = sub.add_parser("embed", help="embed text through the cache")
    embed.add_argument("text", nargs="+")
    embed.add_argument("--model", default="demo", help="model identity for cache keys")
    embed.add_argument("--dim", type=int, default=64, help="dimension of the demo embedder")
    embed.set_defaults(func=cmd_embed)

    stats = sub.add_parser("stats", help="show what is stored")
    stats.add_argument("--model", default=None, help="restrict to one model")
    stats.set_defaults(func=cmd_stats)

    purge = sub.add_parser("purge", help="delete entries")
    purge.add_argument("--model", default=None, help="restrict to one model")
    purge.set_defaults(func=cmd_purge)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (StoreError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
