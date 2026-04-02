#!/usr/bin/env python3
"""Resolve the smallest safe Bazel target set for clang-tidy."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FULL_RUN_FILES = {
    ".bazelrc",
    ".clang-tidy",
    "BUILD",
    "WORKSPACE",
    "envpool/BUILD",
    "envpool/pip.bzl",
    "envpool/requirements.bzl",
    "envpool/workspace0.bzl",
    "envpool/workspace1.bzl",
}
FULL_RUN_PREFIXES = ("third_party/",)
CPP_SUFFIXES = (".cc", ".h")
CC_RULE_KIND = "cc_(library|test)"
SKIP_TARGETS = frozenset((
    "//envpool/minigrid_bindings:minigrid_bindings",
))


def _filter_targets(targets: list[str]) -> list[str]:
    return sorted(target for target in targets if target not in SKIP_TARGETS)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _bazel(*args: str) -> str:
    env = os.environ.copy()
    env.setdefault("USE_BAZEL_VERSION", "8.6.0")
    return subprocess.check_output(
        ["bazelisk", *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
    ).strip()


def _is_valid_commit(rev: str) -> bool:
    if not rev or set(rev) == {"0"}:
        return False
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", f"{rev}^{{commit}}"],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        ).returncode
        == 0
    )


def _infer_changed_files() -> list[str]:
    base_sha = os.environ.get("CLANG_TIDY_BASE_SHA", "").strip()
    head_sha = os.environ.get("CLANG_TIDY_HEAD_SHA", "").strip() or "HEAD"
    if not _is_valid_commit(base_sha):
        try:
            base_sha = _git("merge-base", "HEAD", "origin/main")
        except subprocess.CalledProcessError:
            base_sha = _git("rev-parse", "HEAD^")
    committed = _git("diff", "--name-only", base_sha, head_sha)
    working_tree = _git("diff", "--name-only", "HEAD")
    changed_files = {
        line
        for diff in (committed, working_tree)
        for line in diff.splitlines()
        if line
    }
    return sorted(changed_files)


def _resolve_targets(changed_files: list[str]) -> list[str]:
    if not changed_files:
        return []

    for path in changed_files:
        if path in FULL_RUN_FILES or path.startswith(FULL_RUN_PREFIXES):
            query = _bazel("query", f'kind("{CC_RULE_KIND}", //...)')
            return _filter_targets([line for line in query.splitlines() if line])

    package_patterns: set[str] = set()
    for path in changed_files:
        parts = pathlib.PurePosixPath(path).parts
        if len(parts) < 2 or parts[0] != "envpool":
            continue
        if parts[1] == "python":
            continue
        if (
            path.endswith(CPP_SUFFIXES)
            or pathlib.PurePosixPath(path).name == "BUILD"
        ):
            package_patterns.add(f"//envpool/{parts[1]}:*")

    targets: set[str] = set()
    for pattern in sorted(package_patterns):
        query = _bazel("query", f'kind("{CC_RULE_KIND}", {pattern})')
        targets.update(line for line in query.splitlines() if line)
    return _filter_targets(list(targets))


def _main() -> int:
    changed_files = sys.argv[1:] or _infer_changed_files()
    for target in _resolve_targets(changed_files):
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
