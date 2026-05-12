"""Small CLI for local smoke checks."""

from __future__ import annotations

import argparse

from equivcache import CacheConfig
from equivcache.store import SQLiteCacheStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="equivcache")
    parser.add_argument("--db-path", default=str(CacheConfig().db_path))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("stats", help="Show persisted cache entry counts.")
    args = parser.parse_args()

    if args.command == "stats":
        store = SQLiteCacheStore(args.db_path)
        print(f"entries={store.count()} total_hits={store.total_hit_count()}")
        store.close()
