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
"""Regression tests for MuJoCo process-global plugin registration."""

import subprocess
import sys

from absl.testing import absltest


def _assert_imports_succeed(*modules: str) -> None:
    code = "\n".join(f"import {module}" for module in modules)
    subprocess.run([sys.executable, "-c", code], check=True)


class MujocoPluginRegistrationTest(absltest.TestCase):
    """Tests that MuJoCo decoder plugins are safe across extension modules."""

    def test_robotics_then_dmc_imports_share_stl_decoder(self) -> None:
        """Checks the release-test import order that first exposed the issue."""
        _assert_imports_succeed("envpool.mujoco.robotics", "envpool.mujoco.dmc")

    def test_dmc_then_robotics_imports_share_stl_decoder(self) -> None:
        """Checks the opposite import order for standalone DMC users."""
        _assert_imports_succeed("envpool.mujoco.dmc", "envpool.mujoco.robotics")

    def test_robotics_then_metaworld_imports_share_obj_decoder(self) -> None:
        """Checks the release-test import order with MetaWorld linked in."""
        _assert_imports_succeed(
            "envpool.mujoco.robotics", "envpool.mujoco.metaworld"
        )

    def test_metaworld_then_robotics_imports_share_obj_decoder(self) -> None:
        """Checks the opposite import order for standalone MetaWorld users."""
        _assert_imports_succeed(
            "envpool.mujoco.metaworld", "envpool.mujoco.robotics"
        )


if __name__ == "__main__":
    absltest.main()
