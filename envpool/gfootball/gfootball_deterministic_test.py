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
"""Determinism tests for EnvPool gfootball."""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

from envpool.gfootball.gfootball_oracle_util import (
    ALL_TASK_IDS,
    TASK_CONFIG,
    register_gfootball_envs,
)
from envpool.registration import make_gymnasium

register_gfootball_envs()


def _assert_info_equal(
    actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]
) -> None:
    for key in (
        "score",
        "game_mode",
        "ball_owned_team",
        "ball_owned_player",
        "steps_left",
        "engine_seed",
        "episode_number",
        "elapsed_step",
    ):
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


class _GfootballDeterministicTest(absltest.TestCase):
    def run_deterministic_check(self, task_id: str) -> None:
        num_steps = min(32, int(TASK_CONFIG[task_id]["max_episode_steps"]))
        env0 = make_gymnasium(task_id, num_envs=2, seed=7)
        env1 = make_gymnasium(task_id, num_envs=2, seed=7)
        env2 = make_gymnasium(task_id, num_envs=2, seed=8)
        try:
            obs0, info0 = env0.reset()
            obs1, info1 = env1.reset()
            obs2, info2 = env2.reset()
            np.testing.assert_array_equal(obs0, obs1)
            _assert_info_equal(info0, info1)
            differs = not np.array_equal(obs0, obs2) or not np.array_equal(
                info0["engine_seed"], info2["engine_seed"]
            )
            rng = np.random.default_rng(123)
            for _ in range(num_steps):
                action = rng.integers(0, 19, size=2, dtype=np.int32)
                step0 = env0.step(action)
                step1 = env1.step(action)
                step2 = env2.step(action)
                for actual, expected in zip(
                    step0[:-1], step1[:-1], strict=True
                ):
                    np.testing.assert_array_equal(actual, expected)
                _assert_info_equal(step0[-1], step1[-1])
                differs = differs or any(
                    not np.array_equal(actual, other)
                    for actual, other in zip(
                        step0[:-1], step2[:-1], strict=True
                    )
                )
                differs = differs or any(
                    not np.array_equal(step0[-1][key], step2[-1][key])
                    for key in (
                        "score",
                        "game_mode",
                        "ball_owned_team",
                        "ball_owned_player",
                        "steps_left",
                        "engine_seed",
                        "episode_number",
                    )
                )
            self.assertTrue(
                differs, msg=f"expected different rollout for {task_id}"
            )
        finally:
            env0.close()
            env1.close()
            env2.close()

    def test_all_registered_tasks_are_deterministic(self) -> None:
        for task_id in ALL_TASK_IDS:
            with self.subTest(task_id=task_id):
                self.run_deterministic_check(task_id)


if __name__ == "__main__":
    absltest.main()
