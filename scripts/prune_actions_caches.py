#!/usr/bin/env python3

"""Keep only the newest GitHub Actions caches for a given key prefix."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any


def _github_request(
    method: str, url: str, token: str
) -> dict[str, Any] | list[dict[str, Any]] | None:
    request = urllib.request.Request(
        url,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 404}:
            message = exc.read().decode("utf-8", errors="replace")
            print(
                f"Skipping cache pruning for {url}: HTTP {exc.code} {message}",
                file=sys.stderr,
            )
            return None
        raise
    if not payload:
        return None
    return json.loads(payload)


def _delete_cache(repo: str, cache_id: int, token: str) -> bool:
    url = f"https://api.github.com/repos/{repo}/actions/caches/{cache_id}"
    request = urllib.request.Request(
        url,
        method="DELETE",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request):
            return True
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 404}:
            message = exc.read().decode("utf-8", errors="replace")
            print(
                f"Skipping cache deletion for {cache_id}: "
                f"HTTP {exc.code} {message}",
                file=sys.stderr,
            )
            return False
        raise


def _list_matching_caches(
    repo: str, prefix: str, token: str
) -> list[dict[str, Any]]:
    caches: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"per_page": 100, "page": page})
        url = f"https://api.github.com/repos/{repo}/actions/caches?{query}"
        response = _github_request("GET", url, token)
        if response is None:
            return []
        page_entries = response.get("actions_caches", [])
        caches.extend(
            entry for entry in page_entries if entry["key"].startswith(prefix)
        )
        if len(page_entries) < 100:
            return caches
        page += 1


def _sort_key(entry: dict[str, Any]) -> tuple[datetime, int]:
    created_at = entry.get("created_at") or "1970-01-01T00:00:00Z"
    try:
        timestamp = datetime.fromisoformat(created_at)
    except ValueError:
        timestamp = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return timestamp, int(entry["id"])


def main() -> int:
    """Delete stale caches so only the newest entries per prefix remain."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--keep", type=int, default=1)
    args = parser.parse_args()

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GH_TOKEN or GITHUB_TOKEN is required", file=sys.stderr)
        return 1

    caches = sorted(
        _list_matching_caches(args.repo, args.prefix, token),
        key=_sort_key,
        reverse=True,
    )
    stale_caches = caches[args.keep :]
    if not stale_caches:
        print(f"No stale caches found for prefix {args.prefix}")
        return 0

    for cache in stale_caches:
        cache_id = cache["id"]
        cache_key = cache["key"]
        if _delete_cache(args.repo, cache_id, token):
            print(f"Deleted cache {cache_id} ({cache_key})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
