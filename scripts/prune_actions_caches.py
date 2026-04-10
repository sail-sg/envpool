#!/usr/bin/env python3

"""Prune stale GitHub Actions caches for a given key prefix."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
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


def _branch_exists(repo: str, branch: str, token: str) -> bool:
    quoted_branch = urllib.parse.quote(branch, safe="")
    url = f"https://api.github.com/repos/{repo}/branches/{quoted_branch}"
    request = urllib.request.Request(
        url,
        method="GET",
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
        if exc.code == 404:
            return False
        if exc.code == 403:
            message = exc.read().decode("utf-8", errors="replace")
            print(
                f"Could not check whether branch {branch!r} exists: "
                f"HTTP {exc.code} {message}. Keeping its caches.",
                file=sys.stderr,
            )
            return True
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


def _branch_name_from_cache_key(key: str, prefix: str) -> str | None:
    if not key.startswith(prefix):
        return None
    suffix = key[len(prefix) :]
    branch, separator, run_id = suffix.rpartition("-")
    if not separator or not branch or not run_id.isdigit():
        return None
    return branch


def _legacy_stale_caches(
    caches: list[dict[str, Any]], keep: int
) -> list[dict[str, Any]]:
    return sorted(caches, key=_sort_key, reverse=True)[keep:]


def _stale_caches_by_branch(
    repo: str,
    caches: list[dict[str, Any]],
    prefix: str,
    keep: int,
    delete_missing_branches: bool,
    token: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ungrouped: list[dict[str, Any]] = []
    for cache in caches:
        branch = _branch_name_from_cache_key(cache["key"], prefix)
        if branch is None:
            ungrouped.append(cache)
            continue
        grouped[branch].append(cache)

    stale_caches: dict[int, dict[str, Any]] = {}
    branch_exists_cache: dict[str, bool] = {}
    for branch, branch_caches in grouped.items():
        if delete_missing_branches:
            branch_exists_cache[branch] = _branch_exists(repo, branch, token)
            if not branch_exists_cache[branch]:
                print(
                    f"Branch {branch!r} no longer exists; deleting its caches"
                )
                for cache in branch_caches:
                    stale_caches[int(cache["id"])] = cache
                continue

        sorted_branch_caches = sorted(
            branch_caches, key=_sort_key, reverse=True
        )
        for cache in sorted_branch_caches[keep:]:
            stale_caches[int(cache["id"])] = cache

    if ungrouped:
        print(
            f"Skipping {len(ungrouped)} caches whose keys do not match "
            f"'<prefix><branch>-<run_id>' for prefix {prefix}",
            file=sys.stderr,
        )
    return sorted(stale_caches.values(), key=_sort_key, reverse=True)


def main() -> int:
    """Delete stale caches so only the newest entries per prefix remain."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--keep", type=int, default=1)
    parser.add_argument(
        "--group-by-branch",
        action="store_true",
        help="Treat prefix as a suite prefix and keep N caches per branch.",
    )
    parser.add_argument(
        "--delete-missing-branches",
        action="store_true",
        help="Delete matching branch caches when the branch no longer exists.",
    )
    args = parser.parse_args()
    if args.keep < 0:
        print("--keep must be non-negative", file=sys.stderr)
        return 2
    if args.delete_missing_branches and not args.group_by_branch:
        print(
            "--delete-missing-branches requires --group-by-branch",
            file=sys.stderr,
        )
        return 2

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GH_TOKEN or GITHUB_TOKEN is required", file=sys.stderr)
        return 1

    caches = _list_matching_caches(args.repo, args.prefix, token)
    if args.group_by_branch:
        stale_caches = _stale_caches_by_branch(
            args.repo,
            caches,
            args.prefix,
            args.keep,
            args.delete_missing_branches,
            token,
        )
    else:
        stale_caches = _legacy_stale_caches(caches, args.keep)
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
