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
"""Tests for the dm_control render path."""

from typing import Any, cast

import numpy as np
from absl.testing import absltest
from dm_control import suite

import envpool.mujoco.dmc.registration as reg
from envpool.registration import make_gym


def _task_map() -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for domain, task, _ in reg.dmc_mujoco_envs:
        domain_name = "".join(g[:1].upper() + g[1:] for g in domain.split("_"))
        task_name = "".join(g[:1].upper() + g[1:] for g in task.split("_"))
        result[f"{domain_name}{task_name}-v1"] = (domain, task)
    return result


def _render_array(env: Any, env_ids: Any = None) -> np.ndarray:
    frame = env.render(env_ids=env_ids)
    assert frame is not None
    return cast(np.ndarray, frame)


class MujocoDmcRenderTest(absltest.TestCase):
    """Render regression tests for dm_control-backed MuJoCo tasks."""

    def test_rgb_array_render_is_batch_consistent(self) -> None:
        """Batch render output should match per-env renders for WalkerWalk."""
        env = make_gym(
            "WalkerWalk-v1",
            num_envs=2,
            seed=0,
            render_mode="rgb_array",
            render_width=96,
            render_height=72,
        )
        try:
            env.reset()
            frame0 = _render_array(env)
            frame1 = _render_array(env, env_ids=1)
            frames = _render_array(env, env_ids=[0, 1])
            frame0_again = _render_array(env)
            self.assertEqual(frame0.shape, (1, 72, 96, 3))
            self.assertEqual(frame1.shape, (1, 72, 96, 3))
            self.assertEqual(frames.shape, (2, 72, 96, 3))
            self.assertEqual(frames.dtype, np.uint8)
            np.testing.assert_array_equal(frame0[0], frames[0])
            np.testing.assert_array_equal(frame1[0], frames[1])
            np.testing.assert_array_equal(frame0, frame0_again)
        finally:
            env.close()

    def test_render_matches_dm_control_default_camera(self) -> None:
        """Default-camera renders should stay close to dm_control output."""
        threshold = 21.0
        for task_id, (domain, task) in sorted(_task_map().items()):
            with self.subTest(task_id=task_id):
                env = make_gym(
                    task_id,
                    num_envs=1,
                    seed=0,
                    render_mode="rgb_array",
                    render_width=320,
                    render_height=240,
                )
                oracle = suite.load(domain, task, task_kwargs={"random": 0})
                try:
                    env.reset()
                    oracle.reset()
                    frame = _render_array(env)[0].astype(np.int16)
                    expected = np.asarray(
                        oracle.physics.render(
                            height=frame.shape[0],
                            width=frame.shape[1],
                        ),
                        dtype=np.int16,
                    )
                    diff = np.abs(frame - expected).mean()
                    self.assertLess(diff, threshold)
                finally:
                    env.close()
                    close = getattr(oracle, "close", None)
                    if callable(close):
                        close()


if __name__ == "__main__":
    absltest.main()
