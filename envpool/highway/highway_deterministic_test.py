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

_ALL_TASKS = (
    "Exit-v0",
    "Highway-v0",
    "HighwayFast-v0",
    "Intersection-v0",
    "Intersection-v1",
    "IntersectionMultiAgent-v0",
    "IntersectionMultiAgent-v1",
    "LaneKeeping-v0",
    "Merge-v0",
    "Parking-v0",
    "ParkingActionRepeat-v0",
    "ParkingParked-v0",
    "Racetrack-v0",
    "RacetrackLarge-v0",
    "RacetrackOval-v0",
    "Roundabout-v0",
    "TwoWay-v0",
    "UTurn-v0",
)


def _assert_tree_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
        return
    np.testing.assert_array_equal(actual, expected)


def _idle_action(env: Any, task_id: str, num_envs: int) -> np.ndarray:
    if task_id.startswith("IntersectionMultiAgent"):
        return np.ones(2 * num_envs, dtype=np.int32)
    if hasattr(env.action_space, "n"):
        return np.ones(num_envs, dtype=np.int32)
    return np.zeros((num_envs, *env.action_space.shape), env.action_space.dtype)


def _step(env: Any, task_id: str, action: np.ndarray, num_envs: int) -> Any:
    if task_id.startswith("IntersectionMultiAgent"):
        return env.step(action, np.arange(num_envs, dtype=np.int32))
    return env.step(action)


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

    def test_all_registered_highway_tasks_are_deterministic(self) -> None:
        num_envs = 3
        for task_id in _ALL_TASKS:
            with self.subTest(task_id=task_id):
                num_envs = (
                    1 if task_id.startswith("IntersectionMultiAgent") else 3
                )
                env0 = make_gymnasium(task_id, num_envs=num_envs, seed=7)
                env1 = make_gymnasium(task_id, num_envs=num_envs, seed=7)
                try:
                    obs0, _ = env0.reset()
                    obs1, _ = env1.reset()
                    _assert_tree_equal(obs0, obs1)
                    action = _idle_action(env0, task_id, num_envs)
                    for _ in range(10):
                        step0 = _step(env0, task_id, action, num_envs)
                        step1 = _step(env1, task_id, action, num_envs)
                        for actual, expected in zip(
                            step0[:-1], step1[:-1], strict=True
                        ):
                            _assert_tree_equal(actual, expected)
                        _assert_tree_equal(step0[-1], step1[-1])
                finally:
                    env0.close()
                    env1.close()


if __name__ == "__main__":
    absltest.main()
