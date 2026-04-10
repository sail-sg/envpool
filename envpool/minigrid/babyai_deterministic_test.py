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
"""Determinism checks for the C++ BabyAI backend."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from absl.testing import absltest

import envpool.minigrid.registration  # noqa: F401
from envpool.minigrid.babyai_test_utils import babyai_task_ids
from envpool.registration import make_gym


def _max_debug_steps(env: Any, num_envs: int) -> int:
    states = cast(Any, env)._debug_states(np.arange(num_envs, dtype=np.int32))
    return max(int(state.max_steps) for state in states)


class BabyAIEnvPoolDeterministicTest(absltest.TestCase):
    """Determinism checks for same-seed and different-seed BabyAI rollouts."""

    def _obs_from_reset(
        self,
        reset_out: tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
        | dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        if isinstance(reset_out, tuple):
            return reset_out[0]
        return reset_out

    def _assert_obs_equal(
        self,
        obs0: dict[str, np.ndarray],
        obs1: dict[str, np.ndarray],
    ) -> None:
        for key in obs0:
            np.testing.assert_array_equal(
                obs0[key],
                obs1[key],
                err_msg=f"obs[{key}]",
            )

    def _run_deterministic_check(
        self,
        task_id: str,
        num_envs: int = 4,
        total: int | None = None,
        action_seed: int = 1,
        **kwargs: Any,
    ) -> None:
        env0 = make_gym(task_id, num_envs=num_envs, seed=0, **kwargs)
        env1 = make_gym(task_id, num_envs=num_envs, seed=0, **kwargs)
        act_space = env0.action_space
        act_space.seed(action_seed)
        try:
            self._assert_obs_equal(
                self._obs_from_reset(env0.reset()),
                self._obs_from_reset(env1.reset()),
            )
            if total is None:
                total = _max_debug_steps(env0, num_envs)
            for _ in range(total):
                action = np.array([act_space.sample() for _ in range(num_envs)])
                obs0, rew0, term0, trunc0, info0 = env0.step(action)
                obs1, rew1, term1, trunc1, info1 = env1.step(action)
                self._assert_obs_equal(obs0, obs1)
                np.testing.assert_allclose(rew0, rew1)
                np.testing.assert_array_equal(term0, term1)
                np.testing.assert_array_equal(trunc0, trunc1)
                np.testing.assert_array_equal(
                    info0["elapsed_step"],
                    info1["elapsed_step"],
                )
                np.testing.assert_array_equal(
                    info0["agent_pos"],
                    info1["agent_pos"],
                )
        finally:
            env0.close()
            env1.close()

    def _run_different_seed_check(
        self,
        task_id: str,
        num_envs: int = 4,
        total: int | None = None,
        action_seed: int = 1,
        **kwargs: Any,
    ) -> None:
        env0 = make_gym(task_id, num_envs=num_envs, seed=0, **kwargs)
        env1 = make_gym(task_id, num_envs=num_envs, seed=1, **kwargs)
        act_space = env0.action_space
        act_space.seed(action_seed)
        try:
            obs0 = self._obs_from_reset(env0.reset())
            obs1 = self._obs_from_reset(env1.reset())
            if total is None:
                total = _max_debug_steps(env0, num_envs)
            differs = any(
                not np.array_equal(obs0[key], obs1[key]) for key in obs0
            )
            for _ in range(total):
                action = np.array([act_space.sample() for _ in range(num_envs)])
                obs0, rew0, term0, trunc0, _ = env0.step(action)
                obs1, rew1, term1, trunc1, _ = env1.step(action)
                differs = (
                    differs
                    or any(
                        not np.array_equal(obs0[key], obs1[key]) for key in obs0
                    )
                    or not np.allclose(rew0, rew1)
                    or not np.array_equal(term0, term1)
                    or not np.array_equal(trunc0, trunc1)
                )
                if differs:
                    break
            self.assertTrue(
                differs,
                msg=f"expected different rollouts for {task_id}",
            )
        finally:
            env0.close()
            env1.close()

    def test_registered_babyai_envs_same_seed(self) -> None:
        """Same seed should produce identical rollouts for all BabyAI tasks."""
        for task_id in babyai_task_ids():
            with self.subTest(task_id=task_id):
                print(f"deterministic {task_id}", flush=True)
                self._run_deterministic_check(task_id)

    def test_randomized_envs_different_seed(self) -> None:
        """Different seeds should eventually diverge for randomized tasks."""
        for task_id in [
            "BabyAI-BossLevel-v0",
            "BabyAI-GoToObj-v0",
            "BabyAI-GoToObjMazeS4R2-v0",
            "BabyAI-MiniBossLevel-v0",
            "BabyAI-OpenDoorsOrderN4-v0",
            "BabyAI-PickupLoc-v0",
            "BabyAI-PutNextS7N4-v0",
            "BabyAI-SynthSeq-v0",
            "BabyAI-UnblockPickup-v0",
            "BabyAI-UnlockToUnlock-v0",
        ]:
            with self.subTest(task_id=task_id):
                self._run_different_seed_check(task_id)


if __name__ == "__main__":
    absltest.main()
