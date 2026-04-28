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
"""Smoke tests for staged MyoSuite MuJoCo asset loadability."""

from __future__ import annotations

from pathlib import Path

import mujoco
from absl.testing import absltest

from envpool.mujoco.myosuite.metadata import MYOSUITE_DIRECT_ENTRIES
from envpool.mujoco.myosuite.paths import myosuite_asset_root


def _unique_model_paths() -> tuple[str, ...]:
    return tuple(
        sorted({
            entry["kwargs"]["model_path"] for entry in MYOSUITE_DIRECT_ENTRIES
        })
    )


def _unique_reference_paths() -> tuple[str, ...]:
    return tuple(
        sorted({
            entry["kwargs"]["reference"]
            for entry in MYOSUITE_DIRECT_ENTRIES
            if isinstance(entry["kwargs"].get("reference"), str)
        })
    )


def _loadable_model_paths() -> tuple[str, ...]:
    return tuple(
        sorted({
            entry["kwargs"]["model_path"]
            for entry in MYOSUITE_DIRECT_ENTRIES
            if not entry["model_placeholders"]
        })
    )


class MyoSuiteAssetSmokeTest(absltest.TestCase):
    """Verifies the staged MyoSuite model tree is usable from runfiles."""

    def test_all_registered_model_paths_exist(self) -> None:
        """Checks that every generated direct task model path is staged."""
        root = myosuite_asset_root()
        for relative_path in _unique_model_paths():
            with self.subTest(relative_path=relative_path):
                self.assertTrue((root / relative_path).exists())

    def test_reference_motion_files_exist(self) -> None:
        """Checks that generated MyoDM motion references are staged."""
        root = myosuite_asset_root()
        for relative_path in _unique_reference_paths():
            with self.subTest(relative_path=relative_path):
                self.assertTrue((root / relative_path).exists())

    def test_loadable_registered_models_open_in_mujoco(self) -> None:
        """Checks that standalone staged models load through MuJoCo."""
        root = myosuite_asset_root()
        for relative_path in _loadable_model_paths():
            with self.subTest(relative_path=relative_path):
                xml_path = root / relative_path
                self.assertIsInstance(xml_path, Path)
                model = mujoco.MjModel.from_xml_path(str(xml_path))
                self.assertGreater(model.nq, 0, relative_path)
                self.assertGreater(model.nv, 0, relative_path)


if __name__ == "__main__":
    absltest.main()
