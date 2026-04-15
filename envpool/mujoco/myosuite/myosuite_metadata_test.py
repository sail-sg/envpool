# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regression tests for generated MyoSuite upstream surface metadata."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from absl.testing import absltest

from envpool.mujoco.myosuite.metadata import (
    MYOSUITE_COUNTS,
    MYOSUITE_DIRECT_IDS,
    MYOSUITE_EXPANDED_IDS,
    MYOSUITE_NOTES,
    MYOSUITE_PINS,
    load_myosuite_metadata,
)
from envpool.mujoco.myosuite.paths import resolve_workspace_path


def _find_vendored_myosuite_upstream_root() -> Path:
    """Locate the vendored upstream source tree in Bazel runfiles."""
    roots = [Path.cwd()]
    for env_name in ("RUNFILES_DIR", "TEST_SRCDIR"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))
    for root in roots:
        direct = root / "myosuite_src"
        if (direct / "myosuite/envs/myo/myobase/__init__.py").exists():
            return direct
        for init_path in root.rglob("myosuite/envs/myo/myobase/__init__.py"):
            candidate = init_path.parents[4]
            if (candidate / "pyproject.toml").exists():
                return candidate
    raise FileNotFoundError("Unable to locate vendored myosuite upstream root")


class MyoSuiteMetadataTest(absltest.TestCase):
    """Checks the vendored upstream registry snapshot stays stable."""

    def test_upstream_pin_is_frozen(self) -> None:
        """Checks the checked-in metadata stays anchored to the chosen release."""
        self.assertEqual(
            MYOSUITE_PINS["myosuite"],
            {
                "version": "v2.11.6",
                "commit": "05cb84678373f91271004f99602ebbf01e57d1a1",
            },
        )

    def test_surface_counts_match_generated_snapshot(self) -> None:
        """Checks the generated direct and expanded ID counts remain stable."""
        self.assertEqual(
            MYOSUITE_COUNTS,
            {
                "direct_total": 244,
                "expanded_total": 358,
                "myobase_direct": 35,
                "myochallenge_direct": 19,
                "myodm_direct": 190,
            },
        )
        self.assertLen(MYOSUITE_DIRECT_IDS, MYOSUITE_COUNTS["direct_total"])
        self.assertLen(MYOSUITE_EXPANDED_IDS, MYOSUITE_COUNTS["expanded_total"])

    def test_surface_lists_are_sorted_and_unique(self) -> None:
        """Checks the generated ID lists are already canonicalized."""
        self.assertEqual(list(MYOSUITE_DIRECT_IDS), sorted(MYOSUITE_DIRECT_IDS))
        self.assertEqual(
            list(MYOSUITE_EXPANDED_IDS), sorted(MYOSUITE_EXPANDED_IDS)
        )
        self.assertLen(set(MYOSUITE_DIRECT_IDS), len(MYOSUITE_DIRECT_IDS))
        self.assertLen(set(MYOSUITE_EXPANDED_IDS), len(MYOSUITE_EXPANDED_IDS))

    def test_expected_representative_ids_are_present(self) -> None:
        """Checks representative direct and expanded IDs are preserved."""
        expected_direct = {
            "myoElbowPose1D6MRandom-v0",
            "myoArmReachRandom-v0",
            "myoChallengeBaodingP2-v1",
            "myoChallengeTableTennisP2-v0",
            "MyoHandBananaPass-v0",
            "MyoHandWaterbottleShake-v0",
        }
        expected_expanded_only = {
            "myoSarcElbowPose1D6MRandom-v0",
            "myoFatiElbowPose1D6MRandom-v0",
            "myoReafHandPoseRandom-v0",
        }
        self.assertTrue(expected_direct.issubset(MYOSUITE_DIRECT_IDS))
        self.assertTrue(expected_direct.issubset(MYOSUITE_EXPANDED_IDS))
        self.assertTrue(expected_expanded_only.issubset(MYOSUITE_EXPANDED_IDS))

    def test_duplicate_note_matches_upstream_myoedits_overlap(self) -> None:
        """Checks duplicate-note metadata tracks the upstream myoedits overlap."""
        self.assertEqual(
            MYOSUITE_NOTES["myoedits_duplicate_ids"],
            [
                "myoArmReachFixed-v0",
                "myoArmReachRandom-v0",
            ],
        )

    def test_checked_in_metadata_matches_vendored_upstream_sources(
        self,
    ) -> None:
        """Checks the checked-in snapshot still matches the pinned upstream."""
        upstream_root = _find_vendored_myosuite_upstream_root()
        generator = resolve_workspace_path(
            "third_party/myosuite/generate_metadata.py"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "generated_env_ids.json"
            subprocess.run(
                [
                    sys.executable,
                    str(generator),
                    "--upstream-root",
                    str(upstream_root),
                    "--version",
                    "v2.11.6",
                    "--commit",
                    "05cb84678373f91271004f99602ebbf01e57d1a1",
                    "--output",
                    str(output_path),
                ],
                check=True,
            )
            regenerated = json.loads(output_path.read_text())
        self.assertEqual(regenerated, load_myosuite_metadata())


if __name__ == "__main__":
    absltest.main()
