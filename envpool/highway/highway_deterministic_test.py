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
"""Determinism tests for Highway environments."""

from __future__ import annotations

from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.highway.registration  # noqa: F401
from envpool.registration import make_gymnasium


class _HighwayDeterministicTest(absltest.TestCase):
    def run_deterministic_check(
        self,
        task_id: str,
        num_envs: int = 4,
        num_steps: int = 300,
        **kwargs: Any,
    ) -> None:
        env0 = make_gymnasium(task_id, num_envs=num_envs, seed=0, **kwargs)
        env1 = make_gymnasium(task_id, num_envs=num_envs, seed=0, **kwargs)
        env2 = make_gymnasium(task_id, num_envs=num_envs, seed=1, **kwargs)
        try:
            rng = np.random.default_rng(123)
            env0.reset()
            env1.reset()
            env2.reset()
            for _ in range(num_steps):
                action = rng.integers(0, env0.action_space.n, size=num_envs)
                obs0, rew0, term0, trunc0, info0 = env0.step(action)
                obs1, rew1, term1, trunc1, info1 = env1.step(action)
                obs2, rew2, term2, trunc2, info2 = env2.step(action)

                np.testing.assert_allclose(obs0, obs1)
                np.testing.assert_allclose(rew0, rew1)
                np.testing.assert_array_equal(term0, term1)
                np.testing.assert_array_equal(trunc0, trunc1)
                np.testing.assert_allclose(info0["speed"], info1["speed"])

                self.assertFalse(np.allclose(obs0, obs2))
                self.assertTrue(np.all(np.isfinite(obs0)))
                self.assertTrue(np.all(np.isfinite(obs2)))
        finally:
            env0.close()
            env1.close()
            env2.close()

    def test_highway(self) -> None:
        self.run_deterministic_check("Highway-v0")
        self.run_deterministic_check("Highway-v0", max_episode_steps=4)

    def test_highway_fast(self) -> None:
        self.run_deterministic_check("HighwayFast-v0")

    def test_highway_config_variants(self) -> None:
        self.run_deterministic_check(
            "Highway-v0",
            num_steps=120,
            lanes_count=2,
            vehicles_count=8,
            initial_lane_id=0,
        )
        self.run_deterministic_check(
            "Highway-v0",
            num_steps=120,
            lanes_count=4,
            vehicles_count=8,
            initial_lane_id=2,
            simulation_frequency=10,
            policy_frequency=2,
        )
        self.run_deterministic_check(
            "HighwayFast-v0",
            num_steps=120,
            lanes_count=5,
            vehicles_count=12,
            initial_lane_id=-1,
            vehicles_density=0.5,
        )


if __name__ == "__main__":
    absltest.main()
