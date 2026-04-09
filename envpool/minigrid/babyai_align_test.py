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
"""Alignment tests for the C++ BabyAI backend."""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np
from absl.testing import absltest

from envpool.minigrid.babyai_test_utils import (
    babyai_task_ids,
    debug_state,
    mission_from_obs,
    patch_verifier_state,
)
from envpool.registration import make_gymnasium


class BabyAIEnvPoolAlignTest(absltest.TestCase):
    """Alignment checks against upstream BabyAI environments."""

    def _check_spec(
        self,
        spec0: gym.spaces.Space,
        spec1: gym.spaces.Space,
    ) -> None:
        self.assertEqual(spec0.dtype, spec1.dtype)
        if isinstance(spec0, gym.spaces.Discrete):
            assert isinstance(spec1, gym.spaces.Discrete)
            self.assertEqual(spec0.n, spec1.n)
        elif isinstance(spec0, gym.spaces.Box):
            assert isinstance(spec1, gym.spaces.Box)
            np.testing.assert_allclose(spec0.low, spec1.low)
            np.testing.assert_allclose(spec0.high, spec1.high)

    def _run_align_check(
        self,
        task_id: str,
        total: int = 100,
        **kwargs: Any,
    ) -> None:
        oracle_env = gym.make(task_id)
        env = make_gymnasium(task_id, num_envs=1, seed=0, **kwargs)
        try:
            oracle_obs_space = cast(Any, oracle_env.observation_space)
            self._check_spec(
                oracle_obs_space["direction"],
                env.observation_space["direction"],
            )
            self._check_spec(
                oracle_obs_space["image"],
                env.observation_space["image"],
            )
            self._check_spec(oracle_env.action_space, env.action_space)

            obs, info = env.reset()
            oracle_env.reset(seed=0)
            patch_verifier_state(
                cast(Any, oracle_env.unwrapped),
                task_id,
                debug_state(env),
                int(info["elapsed_step"][0]),
            )
            self.assertEqual(
                cast(Any, oracle_env.unwrapped).mission,
                mission_from_obs(obs),
            )

            done = False
            for _ in range(total):
                act = oracle_env.action_space.sample()
                obs, rew, term, trunc, info = env.step(np.array([act]))
                if done:
                    oracle_env.reset()
                    patch_verifier_state(
                        cast(Any, oracle_env.unwrapped),
                        task_id,
                        debug_state(env),
                        int(info["elapsed_step"][0]),
                    )
                    done = bool(term[0] or trunc[0])
                    continue

                (
                    oracle_obs,
                    oracle_rew,
                    oracle_term,
                    oracle_trunc,
                    _,
                ) = cast(Any, oracle_env.step(act))
                np.testing.assert_array_equal(
                    oracle_obs["direction"],
                    obs["direction"][0],
                )
                np.testing.assert_array_equal(
                    oracle_obs["image"],
                    obs["image"][0],
                )
                self.assertEqual(oracle_obs["mission"], mission_from_obs(obs))
                np.testing.assert_allclose(
                    float(oracle_rew),
                    float(rew[0]),
                    rtol=1e-6,
                )
                self.assertEqual(
                    bool(oracle_term and not oracle_trunc),
                    bool(term[0]),
                )
                self.assertEqual(bool(oracle_trunc), bool(trunc[0]))
                np.testing.assert_array_equal(
                    np.asarray(cast(Any, oracle_env.unwrapped).agent_pos),
                    info["agent_pos"][0],
                )
                done = bool(oracle_term or oracle_trunc)
        finally:
            oracle_env.close()
            env.close()

    def test_registered_babyai_envs(self) -> None:
        """All registered BabyAI tasks should match upstream transitions."""
        for task_id in babyai_task_ids():
            with self.subTest(task_id=task_id):
                print(f"align {task_id}", flush=True)
                self._run_align_check(task_id)


if __name__ == "__main__":
    absltest.main()
