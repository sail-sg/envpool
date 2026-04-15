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

from envpool.mujoco.myosuite.paths import myosuite_asset_root

_TOP_LEVEL_MODEL_XMLS = (
    "envs/myo/assets/arm/myoarm_bionic_bimanual.xml",
    "envs/myo/assets/arm/myoarm_relocate.xml",
    "envs/myo/assets/arm/myoarm_tabletennis.xml",
    "envs/myo/assets/elbow/myoelbow_1dof6muscles.xml",
    "envs/myo/assets/elbow/myoelbow_1dof6muscles_1dofexo.xml",
    "envs/myo/assets/hand/myohand_baoding.xml",
    "envs/myo/assets/hand/myohand_die.xml",
    "envs/myo/assets/hand/myohand_hold.xml",
    "envs/myo/assets/hand/myohand_keyturn.xml",
    "envs/myo/assets/hand/myohand_pen.xml",
    "envs/myo/assets/hand/myohand_pose.xml",
    "envs/myo/assets/hand/myohand_sar.xml",
    "envs/myo/assets/hand/myohand_tabletop.xml",
    "envs/myo/assets/leg/myolegs_chasetag.xml",
    "envs/myo/assets/leg/myoosl_runtrack.xml",
    "envs/myo/assets/leg_soccer/myolegs_soccer.xml",
)


class MyoSuiteAssetSmokeTest(absltest.TestCase):
    """Verifies the staged MyoSuite model tree is usable from runfiles."""

    def test_expected_top_level_models_exist(self) -> None:
        root = myosuite_asset_root()
        for relative_path in _TOP_LEVEL_MODEL_XMLS:
            with self.subTest(relative_path=relative_path):
                self.assertTrue((root / relative_path).exists())

    def test_expected_top_level_models_load_in_mujoco(self) -> None:
        root = myosuite_asset_root()
        for relative_path in _TOP_LEVEL_MODEL_XMLS:
            with self.subTest(relative_path=relative_path):
                xml_path = root / relative_path
                self.assertIsInstance(xml_path, Path)
                model = mujoco.MjModel.from_xml_path(str(xml_path))
                self.assertGreater(model.nq, 0, relative_path)
                self.assertGreater(model.nv, 0, relative_path)


if __name__ == "__main__":
    absltest.main()
