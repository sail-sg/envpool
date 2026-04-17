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
from collections import Counter
from pathlib import Path

from absl.testing import absltest

from envpool.mujoco.myosuite.config import resolve_myosuite_task_config
from envpool.mujoco.myosuite.metadata import (
    MYOSUITE_COUNTS,
    MYOSUITE_DIRECT_ENTRIES,
    MYOSUITE_DIRECT_ENTRY_BY_ID,
    MYOSUITE_DIRECT_IDS,
    MYOSUITE_EXPANDED_IDS,
    MYOSUITE_NOTES,
    MYOSUITE_PINS,
    MYOSUITE_SUITES,
    load_myosuite_metadata,
)
from envpool.mujoco.myosuite.paths import (
    myosuite_asset_root,
    resolve_workspace_path,
)


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
                "direct_total": 254,
                "expanded_total": 398,
                "myobase_direct": 45,
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
            "myoHandPose7Fixed-v0",
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

    def test_direct_entries_cover_direct_and_expanded_ids(self) -> None:
        """Checks direct-entry metadata stays aligned with the ID surfaces."""
        self.assertLen(MYOSUITE_DIRECT_ENTRIES, MYOSUITE_COUNTS["direct_total"])
        self.assertEqual(
            [entry["id"] for entry in MYOSUITE_DIRECT_ENTRIES],
            list(MYOSUITE_DIRECT_IDS),
        )
        expanded = sorted(
            {entry["id"] for entry in MYOSUITE_DIRECT_ENTRIES}
            | {
                variant["variant_id"]
                for entry in MYOSUITE_DIRECT_ENTRIES
                for variant in entry["variant_defs"]
            }
        )
        self.assertEqual(expanded, list(MYOSUITE_EXPANDED_IDS))

    def test_representative_direct_entries_are_stable(self) -> None:
        """Checks representative task definitions remain pinned."""
        myobase_reach = MYOSUITE_DIRECT_ENTRY_BY_ID["myoArmReachRandom-v0"]
        self.assertEqual(myobase_reach["suite"], "myobase")
        self.assertEqual(myobase_reach["registration_suite"], "myoedits")
        self.assertEqual(myobase_reach["class_name"], "ReachEnvV0")
        self.assertEqual(
            myobase_reach["kwargs"]["model_path"],
            "simhive/myo_sim/arm/myoarm.xml",
        )
        self.assertEqual(myobase_reach["kwargs"]["far_th"], 1.0)
        self.assertEqual(
            myobase_reach["kwargs"]["target_reach_range"]["IFtip"],
            [[-0.35, -0.42, 0.98], [0.0, -0.07, 1.83]],
        )
        self.assertEqual(
            myobase_reach["kwargs"]["edit_fn"], "edit_fn_arm_reaching"
        )
        self.assertEqual(
            myobase_reach["default_config"]["obs_dim"],
            80,
        )
        self.assertEqual(
            myobase_reach["default_config"]["model_path"],
            "simhive/myo_sim/arm/myoarm_reach.xml",
        )
        self.assertEqual(
            myobase_reach["default_config"]["action_dim"],
            34,
        )
        self.assertEqual(
            [
                variant["variant_id"]
                for variant in myobase_reach["variant_defs"]
            ],
            [
                "myoFatiArmReachRandom-v0",
                "myoSarcArmReachRandom-v0",
            ],
        )

        myodm_track = MYOSUITE_DIRECT_ENTRY_BY_ID["MyoHandAirplaneFly-v0"]
        self.assertEqual(myodm_track["suite"], "myodm")
        self.assertEqual(myodm_track["class_name"], "TrackEnv")
        self.assertEqual(
            myodm_track["kwargs"]["model_path"],
            "envs/myo/assets/hand/myohand_object.xml",
        )
        self.assertEqual(
            myodm_track["kwargs"]["reference"],
            "envs/myo/myodm/data/MyoHand_airplane_fly1.npz",
        )
        self.assertEqual(myodm_track["model_placeholders"], ["OBJECT_NAME"])
        self.assertEqual(myodm_track["default_config"]["robot_dim"], 29)
        self.assertEqual(myodm_track["default_config"]["object_dim"], 7)
        self.assertEqual(myodm_track["default_config"]["obs_dim"], 142)
        self.assertEmpty(myodm_track["variant_defs"])

    def test_reach_default_configs_capture_derived_render_flags(self) -> None:
        """Checks baked reach configs keep walk-specific derived flags."""
        walk_reach = MYOSUITE_DIRECT_ENTRY_BY_ID["myoLegStandRandom-v0"]
        self.assertEqual(
            walk_reach["entry_module"],
            "myosuite.envs.myo.myobase.walk_v0",
        )
        self.assertEqual(walk_reach["class_name"], "ReachEnvV0")
        self.assertTrue(walk_reach["default_config"]["target_pos_relative_to_tip"])
        self.assertTrue(walk_reach["default_config"]["hide_skin_geom_group_1"])
        self.assertTrue(walk_reach["default_config"]["hide_terrain"])
        self.assertEqual(
            walk_reach["default_config"]["joint_random_range"],
            [-0.2, 0.2],
        )

        arm_reach = MYOSUITE_DIRECT_ENTRY_BY_ID["myoArmReachRandom-v0"]
        self.assertEqual(
            arm_reach["entry_module"],
            "myosuite.envs.myo.myobase.reach_v0",
        )
        self.assertEqual(arm_reach["class_name"], "ReachEnvV0")
        self.assertFalse(arm_reach["default_config"]["target_pos_relative_to_tip"])
        self.assertFalse(arm_reach["default_config"]["hide_skin_geom_group_1"])
        self.assertFalse(arm_reach["default_config"]["hide_terrain"])

    def test_suite_breakdowns_match_expected_entrypoint_counts(self) -> None:
        """Checks the pinned suite mix still matches the native port plan."""
        suite_counts = Counter(
            (entry["suite"], entry["class_name"])
            for entry in MYOSUITE_DIRECT_ENTRIES
        )
        self.assertEqual(
            suite_counts,
            Counter({
                ("myobase", "PoseEnvV0"): 20,
                ("myobase", "ReachEnvV0"): 9,
                ("myobase", "TerrainEnvV0"): 3,
                ("myobase", "KeyTurnEnvV0"): 2,
                ("myobase", "TorsoEnvV0"): 2,
                ("myobase", "WalkEnvV0"): 1,
                ("myobase", "ObjHoldFixedEnvV0"): 1,
                ("myobase", "ObjHoldRandomEnvV0"): 1,
                ("myobase", "PenTwirlFixedEnvV0"): 1,
                ("myobase", "PenTwirlRandomEnvV0"): 1,
                ("myobase", "Geometries8EnvV0"): 1,
                ("myobase", "Geometries100EnvV0"): 1,
                ("myobase", "InDistribution"): 1,
                ("myobase", "OutofDistribution"): 1,
                ("myochallenge", "TableTennisEnvV0"): 3,
                ("myochallenge", "RelocateEnvV0"): 3,
                ("myochallenge", "ChaseTagEnvV0"): 3,
                ("myochallenge", "ReorientEnvV0"): 3,
                ("myochallenge", "SoccerEnvV0"): 2,
                ("myochallenge", "RunTrack"): 2,
                ("myochallenge", "BaodingEnvV1"): 2,
                ("myochallenge", "BimanualEnvV1"): 1,
                ("myodm", "TrackEnv"): 190,
            }),
        )

    def test_suite_lists_match_direct_entry_groups(self) -> None:
        """Checks suite helper lists stay consistent with direct entries."""
        self.assertEqual(
            MYOSUITE_SUITES["myobase_direct_ids"],
            sorted(
                entry["id"]
                for entry in MYOSUITE_DIRECT_ENTRIES
                if entry["suite"] == "myobase"
            ),
        )
        self.assertEqual(
            MYOSUITE_SUITES["myochallenge_direct_ids"],
            sorted(
                entry["id"]
                for entry in MYOSUITE_DIRECT_ENTRIES
                if entry["suite"] == "myochallenge"
            ),
        )
        self.assertEqual(
            MYOSUITE_SUITES["myodm_track_ids"],
            sorted(
                entry["id"]
                for entry in MYOSUITE_DIRECT_ENTRIES
                if entry["suite"] == "myodm"
                and isinstance(entry["kwargs"].get("reference"), str)
            ),
        )

    def test_duplicate_note_matches_upstream_myoedits_overlap(self) -> None:
        """Checks duplicate-note metadata tracks the upstream myoedits overlap."""
        self.assertEqual(
            MYOSUITE_NOTES["myoedits_duplicate_ids"],
            [
                "myoArmReachFixed-v0",
                "myoArmReachRandom-v0",
            ],
        )

    def test_generated_paths_exist_in_staged_assets(self) -> None:
        """Checks generated model and reference paths resolve under staged assets."""
        root = myosuite_asset_root()
        for entry in MYOSUITE_DIRECT_ENTRIES:
            with self.subTest(task_id=entry["id"], kind="model"):
                self.assertTrue((root / entry["kwargs"]["model_path"]).exists())
            reference = entry["kwargs"].get("reference")
            if isinstance(reference, str):
                with self.subTest(task_id=entry["id"], kind="reference"):
                    self.assertTrue((root / reference).exists())

    def test_preview_test_overrides_are_filtered_by_supported_spec(self) -> None:
        """Checks pose-only preview state does not leak into torso configs."""
        base_path = str(myosuite_asset_root().parent.parent)

        pose_config = resolve_myosuite_task_config(
            "myoElbowPose1D6MFixed-v0",
            {
                "base_path": base_path,
                "test_target_qpos": [0.123],
            },
        )
        self.assertEqual(pose_config["test_target_qpos"], [0.123])

        torso_config = resolve_myosuite_task_config(
            "myoTorsoPoseFixed-v0",
            {
                "base_path": base_path,
                "test_target_qpos": [0.123],
            },
        )
        self.assertNotIn("test_target_qpos", torso_config)

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
