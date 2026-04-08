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
"""Coverage tests for the upstream highway-env task registry."""

from __future__ import annotations

from typing import Any, NamedTuple

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from absl.testing import absltest
from gymnasium import spaces

import envpool.highway.registration  # noqa: F401
from envpool.registration import make_gymnasium

_UPSTREAM_IDS = frozenset({
    "exit-v0",
    "highway-fast-v0",
    "highway-v0",
    "intersection-multi-agent-v0",
    "intersection-multi-agent-v1",
    "intersection-v0",
    "intersection-v1",
    "lane-keeping-v0",
    "merge-v0",
    "parking-ActionRepeat-v0",
    "parking-parked-v0",
    "parking-v0",
    "racetrack-large-v0",
    "racetrack-oval-v0",
    "racetrack-v0",
    "roundabout-v0",
    "two-way-v0",
    "u-turn-v0",
})


class _Case(NamedTuple):
    official_id: str
    player_count: int = 1
    discrete_action: int = 1
    continuous_action: tuple[float, ...] | None = None
    bridge: bool = True


_CASES = (
    _Case("exit-v0"),
    _Case("highway-fast-v0", bridge=False),
    _Case("highway-v0", bridge=False),
    _Case("intersection-multi-agent-v0", player_count=2),
    _Case("intersection-multi-agent-v1", player_count=2),
    _Case("intersection-v0", discrete_action=1),
    _Case("intersection-v1", continuous_action=(0.0, 0.0)),
    _Case("lane-keeping-v0", continuous_action=(0.0,)),
    _Case("merge-v0"),
    _Case("parking-ActionRepeat-v0", continuous_action=(0.0, 0.0)),
    _Case("parking-parked-v0", continuous_action=(0.0, 0.0)),
    _Case("parking-v0", continuous_action=(0.0, 0.0)),
    _Case("racetrack-large-v0", continuous_action=(0.0,)),
    _Case("racetrack-oval-v0", continuous_action=(0.0,)),
    _Case("racetrack-v0", continuous_action=(0.0,)),
    _Case("roundabout-v0"),
    _Case("two-way-v0"),
    _Case("u-turn-v0"),
)


def _registered_highway_ids() -> set[str]:
    return {
        env_id
        for env_id, spec in gym.envs.registry.items()
        if isinstance(spec.entry_point, str)
        and spec.entry_point.startswith("highway_env.")
    }


def _make_oracle(official_id: str) -> gym.Env:
    constructor_kwargs: dict[str, Any] = {}
    if official_id.startswith("intersection-"):
        constructor_kwargs["config"] = {
            "initial_vehicle_count": 0,
            "spawn_probability": 0.0,
        }
    try:
        wrapper = gym.make(
            official_id, render_mode="rgb_array", **constructor_kwargs
        )
    except TypeError:
        wrapper = gym.make(official_id, **constructor_kwargs)
    env = wrapper.unwrapped
    if official_id.startswith("intersection-"):
        env.configure(env.default_config())
    env.configure({"offscreen_rendering": True, "render_agent": True})
    env.render_mode = "rgb_array"
    return env


def _envpool_action(case: _Case, action_space: spaces.Space[Any]) -> np.ndarray:
    if isinstance(action_space, spaces.Discrete):
        return np.full(case.player_count, case.discrete_action, dtype=np.int32)
    if isinstance(action_space, spaces.Box):
        action = np.asarray(case.continuous_action, dtype=action_space.dtype)
        return np.repeat(action[None, ...], case.player_count, axis=0)
    raise TypeError(f"Unsupported highway action space: {action_space!r}")


def _oracle_action(case: _Case, action_space: spaces.Space[Any]) -> Any:
    if case.player_count != 1:
        return tuple([case.discrete_action] * case.player_count)
    if isinstance(action_space, spaces.Discrete):
        return case.discrete_action
    if isinstance(action_space, spaces.Box):
        return np.asarray(case.continuous_action, dtype=action_space.dtype)
    raise TypeError(f"Unsupported highway action space: {action_space!r}")


def _envpool_step(env: Any, case: _Case, action: np.ndarray) -> tuple[Any, ...]:
    if case.player_count == 1:
        return env.step(action)
    env_id = np.asarray([0], dtype=np.int32)
    return env.step(action, env_id)


def _assert_obs_equal(
    test_case: absltest.TestCase, actual: Any, expected: Any
) -> None:
    if isinstance(expected, tuple):
        expected = np.stack(expected, axis=0)
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        test_case.assertIsInstance(actual, dict)
        test_case.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            np.testing.assert_array_equal(actual[key][0], expected[key])
    else:
        np.testing.assert_array_equal(actual[0], expected)


def _assert_step_equal(
    test_case: absltest.TestCase,
    actual: tuple[Any, Any, np.ndarray, np.ndarray, dict[str, Any]],
    expected: tuple[Any, Any, bool, bool, dict[str, Any]],
) -> None:
    obs, reward, terminated, truncated, _ = actual
    oracle_obs, oracle_reward, oracle_terminated, oracle_truncated, _ = expected
    _assert_obs_equal(test_case, obs, oracle_obs)
    np.testing.assert_array_equal(reward, np.float32(oracle_reward))
    np.testing.assert_array_equal(terminated, np.asarray([oracle_terminated]))
    np.testing.assert_array_equal(truncated, np.asarray([oracle_truncated]))


def _assert_space_compatible(
    test_case: absltest.TestCase,
    envpool_space: spaces.Space[Any],
    official_space: spaces.Space[Any],
) -> None:
    if isinstance(official_space, spaces.Tuple):
        test_case.assertIsInstance(envpool_space, type(official_space[0]))
        test_case.assertEqual(envpool_space.shape, official_space[0].shape)
        test_case.assertEqual(envpool_space.dtype, official_space[0].dtype)
    elif isinstance(official_space, spaces.Dict):
        test_case.assertIsInstance(envpool_space, spaces.Dict)
        test_case.assertEqual(envpool_space.keys(), official_space.keys())
        for key, official_subspace in official_space.items():
            _assert_space_compatible(
                test_case, envpool_space[key], official_subspace
            )
    else:
        test_case.assertIsInstance(envpool_space, type(official_space))
        test_case.assertEqual(envpool_space.shape, official_space.shape)
        test_case.assertEqual(envpool_space.dtype, official_space.dtype)


class _HighwayOfficialCoverageTest(absltest.TestCase):
    def test_upstream_registry_is_fully_listed(self) -> None:
        self.assertEqual(_registered_highway_ids(), _UPSTREAM_IDS)
        self.assertEqual({case.official_id for case in _CASES}, _UPSTREAM_IDS)

    def test_all_upstream_ids_reset_step_and_render_more_than_once(
        self,
    ) -> None:
        for case in _CASES:
            with self.subTest(official_id=case.official_id):
                env = make_gymnasium(
                    case.official_id,
                    num_envs=1,
                    seed=0,
                    render_mode="rgb_array",
                )
                oracle = _make_oracle(case.official_id)
                try:
                    _assert_space_compatible(
                        self,
                        env.observation_space,
                        oracle.observation_space,
                    )
                    action = _envpool_action(case, env.action_space)
                    obs, _ = env.reset()
                    self.assertIsNotNone(obs)
                    frames = []
                    for _ in range(3):
                        obs, reward, terminated, truncated, _ = _envpool_step(
                            env, case, action
                        )
                        self.assertTrue(np.all(np.isfinite(reward)))
                        self.assertEqual(terminated.dtype, np.dtype(bool))
                        self.assertEqual(truncated.dtype, np.dtype(bool))
                        self.assertIsNotNone(obs)
                        frame = env.render()
                        frames.append(frame)
                        self.assertEqual(frame.dtype, np.dtype(np.uint8))
                        self.assertEqual(frame.shape[0], 1)
                        self.assertGreater(frame.shape[1], 0)
                        self.assertGreater(frame.shape[2], 0)
                        self.assertEqual(frame.shape[3], 3)
                        self.assertGreater(int(frame.max()), int(frame.min()))
                    for frame in frames[1:]:
                        self.assertEqual(frame.shape, frames[0].shape)
                finally:
                    env.close()
                    oracle.close()

    def test_official_bridge_tasks_align_bitwise_with_upstream(self) -> None:
        for case in _CASES:
            if not case.bridge:
                continue
            with self.subTest(official_id=case.official_id):
                env = make_gymnasium(
                    case.official_id,
                    num_envs=1,
                    seed=0,
                    render_mode="rgb_array",
                )
                oracle = _make_oracle(case.official_id)
                try:
                    envpool_obs, _ = env.reset()
                    oracle_obs, _ = oracle.reset(seed=0)
                    _assert_obs_equal(self, envpool_obs, oracle_obs)
                    np.testing.assert_array_equal(
                        env.render()[0], oracle.render()
                    )

                    envpool_action = _envpool_action(case, env.action_space)
                    oracle_action = _oracle_action(case, oracle.action_space)
                    for _ in range(2):
                        oracle_step = oracle.step(oracle_action)
                        envpool_step = _envpool_step(env, case, envpool_action)
                        _assert_step_equal(self, envpool_step, oracle_step)
                        np.testing.assert_array_equal(
                            env.render()[0], oracle.render()
                        )
                finally:
                    env.close()
                    oracle.close()


if __name__ == "__main__":
    absltest.main()
