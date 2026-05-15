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
"""EGL teardown regression coverage for native MuJoCo pixel envs."""

from absl.testing import absltest

from envpool.mujoco.pixel_observation_test_utils import (
    assert_egl_async_pixel_envs_do_not_share_context_concurrently,
    assert_egl_pixel_env_teardown_exits_cleanly,
)

_EGL_TEARDOWN_CASES = (
    ("dmc", "envpool.mujoco.dmc.registration", "WalkerWalk-v1"),
    ("gym", "envpool.mujoco.gym.registration", "Walker2d-v4"),
    ("robotics", "envpool.mujoco.robotics.registration", "FetchReach-v4"),
    (
        "metaworld",
        "envpool.mujoco.metaworld.registration",
        "MetaWorld/Reach-v3",
    ),
    (
        "myosuite",
        "envpool.mujoco.myosuite.registration",
        "myoFingerReachFixed-v0",
    ),
    (
        "playground",
        "envpool.mujoco.playground.registration",
        "Go1JoystickFlatTerrain-v1",
    ),
)


class MujocoEglTeardownTest(absltest.TestCase):
    """Teardown tests for MuJoCo env families backed by native GL rendering."""

    def test_pixel_env_teardown_exits_cleanly_for_all_gl_families(self) -> None:
        """All native MuJoCo pixel families should exit cleanly under EGL."""
        assert_egl_pixel_env_teardown_exits_cleanly(self, _EGL_TEARDOWN_CASES)

    def test_async_pixel_envs_do_not_share_egl_context_concurrently(
        self,
    ) -> None:
        """Async pixel envs should not reuse one EGL context concurrently."""
        assert_egl_async_pixel_envs_do_not_share_context_concurrently(self)


if __name__ == "__main__":
    absltest.main()
