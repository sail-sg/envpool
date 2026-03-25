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
"""Determinism checks for the C++ MiniGrid backend."""

from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.minigrid.registration  # noqa: F401
from envpool.registration import list_all_envs, make_gym


class _MiniGridEnvPoolDeterministicTest(absltest.TestCase):
    def minigrid_task_ids(self) -> list[str]:
        task_ids = sorted(
            task_id
            for task_id in list_all_envs()
            if task_id.startswith("MiniGrid-")
        )
        self.assertLen(task_ids, 75)
        return task_ids

    def obs_from_reset(
        self,
        reset_out: tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
        | dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        if isinstance(reset_out, tuple):
            return reset_out[0]
        return reset_out

    def assert_obs_equal(
        self,
        obs0: dict[str, np.ndarray],
        obs1: dict[str, np.ndarray],
    ) -> None:
        for key in obs0:
            np.testing.assert_array_equal(
                obs0[key], obs1[key], err_msg=f"obs[{key}]"
            )

    def run_deterministic_check(
        self,
        task_id: str,
        num_envs: int = 4,
        total: int = 64,
        action_seed: int = 1,
        **kwargs: Any,
    ) -> None:
        env0 = make_gym(task_id, num_envs=num_envs, seed=0, **kwargs)
        env1 = make_gym(task_id, num_envs=num_envs, seed=0, **kwargs)
        act_space = env0.action_space
        act_space.seed(action_seed)
        self.assert_obs_equal(
            self.obs_from_reset(env0.reset()),
            self.obs_from_reset(env1.reset()),
        )
        for _ in range(total):
            action = np.array([act_space.sample() for _ in range(num_envs)])
            self.assert_obs_equal(env0.step(action)[0], env1.step(action)[0])

    def run_different_seed_check(
        self,
        task_id: str,
        num_envs: int = 4,
        total: int = 32,
        action_seed: int = 1,
        **kwargs: Any,
    ) -> None:
        env0 = make_gym(task_id, num_envs=num_envs, seed=0, **kwargs)
        env1 = make_gym(task_id, num_envs=num_envs, seed=1, **kwargs)
        act_space = env0.action_space
        act_space.seed(action_seed)
        obs0 = self.obs_from_reset(env0.reset())
        obs1 = self.obs_from_reset(env1.reset())
        differs = any(not np.array_equal(obs0[key], obs1[key]) for key in obs0)
        for _ in range(total):
            action = np.array([act_space.sample() for _ in range(num_envs)])
            obs0 = env0.step(action)[0]
            obs1 = env1.step(action)[0]
            differs = differs or any(
                not np.array_equal(obs0[key], obs1[key]) for key in obs0
            )
            if differs:
                break
        self.assertTrue(
            differs, msg=f"expected different rollouts for {task_id}"
        )

    def test_registered_minigrid_envs_same_seed(self) -> None:
        for task_id in self.minigrid_task_ids():
            with self.subTest(task_id=task_id):
                self.run_deterministic_check(task_id)

    def test_randomized_envs_different_seed(self) -> None:
        for task_id in [
            "MiniGrid-Empty-Random-5x5-v0",
            "MiniGrid-DoorKey-8x8-v0",
            "MiniGrid-Dynamic-Obstacles-Random-6x6-v0",
            "MiniGrid-Fetch-8x8-N3-v0",
            "MiniGrid-KeyCorridorS6R3-v0",
            "MiniGrid-LockedRoom-v0",
            "MiniGrid-MemoryS17Random-v0",
            "MiniGrid-MultiRoom-N6-v0",
            "MiniGrid-ObstructedMaze-Full-v1",
            "MiniGrid-PutNear-8x8-N3-v0",
        ]:
            with self.subTest(task_id=task_id):
                self.run_different_seed_check(task_id)


if __name__ == "__main__":
    absltest.main()
