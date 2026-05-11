#!/usr/bin/env python3

"""Tests for GitHub Actions cache pruning helpers."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def load_prune_actions_caches():
    """Load the script module from the Bazel runfiles directory."""
    module_path = Path(__file__).with_name("prune_actions_caches.py")
    spec = importlib.util.spec_from_file_location(
        "prune_actions_caches", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prune_actions_caches = load_prune_actions_caches()


class BranchNameFromCacheKeyTest(unittest.TestCase):
    """Verify branch grouping for cache keys."""

    def setUp(self) -> None:
        """Set a stable cache key prefix."""
        self.prefix = "bazel-test-v5-windows-capped-Windows-X64-hash-"

    def branch_name(self, suffix: str) -> str | None:
        """Parse a branch name from a cache key suffix."""
        return prune_actions_caches._branch_name_from_cache_key(
            f"{self.prefix}{suffix}", self.prefix
        )

    def test_legacy_run_id_suffix(self) -> None:
        """Legacy keys use a numeric run id suffix."""
        self.assertEqual(self.branch_name("main-25654277602"), "main")
        self.assertEqual(
            self.branch_name("jiayi/windows-cache-25654277602"),
            "jiayi/windows-cache",
        )
        self.assertEqual(self.branch_name("release-1-25654277602"), "release-1")

    def test_run_attempt_suffix(self) -> None:
        """Rerun-aware keys include a numeric run attempt suffix."""
        self.assertEqual(
            self.branch_name("main-25654277602-attempt2"),
            "main",
        )
        self.assertEqual(
            self.branch_name("jiayi/windows-cache-25654277602-attempt2"),
            "jiayi/windows-cache",
        )
        self.assertEqual(
            self.branch_name("feature-attempt2-25654277602-attempt3"),
            "feature-attempt2",
        )

    def test_malformed_keys(self) -> None:
        """Malformed keys are ignored instead of grouped under a branch."""
        self.assertIsNone(
            prune_actions_caches._branch_name_from_cache_key(
                "other-prefix-main-25654277602", self.prefix
            )
        )
        self.assertIsNone(self.branch_name("main"))
        self.assertIsNone(self.branch_name("main-run"))
        self.assertIsNone(self.branch_name("main-25654277602-attempt"))
        self.assertIsNone(self.branch_name("main-25654277602-attemptx"))


if __name__ == "__main__":
    unittest.main()
