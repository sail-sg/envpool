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
"""Bitwise alignment tests against the official highway-env package."""

from __future__ import annotations

from typing import Any, NamedTuple, cast

import gymnasium as gym
import numpy as np
from absl.testing import absltest
from gymnasium import spaces

from envpool.highway.highway_align_test import _debug_state, _patch_oracle
from envpool.highway.highway_official_coverage_test import _CASES, _Case
from envpool.highway.highway_oracle_util import (
    prepare_official_oracle_import,
    register_highway_envs,
)
from envpool.registration import make_gymnasium

register_highway_envs()
prepare_official_oracle_import()

_OFFICIAL_ALIGN_STEPS = 64
_OFFICIAL_BACKEND_ALIGN_STEPS = 5
_EXIT_ALIGN_STEPS = 6
_INTERSECTION_ALIGN_STEPS = 6
_LANE_KEEPING_ALIGN_STEPS = 2
_RACETRACK_ALIGN_STEPS = 5


class _Step(NamedTuple):
    obs: Any
    reward: Any
    terminated: Any
    truncated: Any
    info: dict[str, Any]


def _make_oracle(official_id: str, config: dict[str, Any] | None = None) -> Any:
    import highway_env  # noqa: F401

    try:
        wrapper = cast(Any, gym.make(official_id, render_mode="rgb_array"))
    except TypeError:
        wrapper = cast(Any, gym.make(official_id))
    env = wrapper.unwrapped
    env.configure({
        "offscreen_rendering": True,
        "render_agent": True,
        **(config or {}),
    })
    env.render_mode = "rgb_array"
    return wrapper


def _official_action(space: spaces.Space[Any], step: int) -> Any:
    if isinstance(space, spaces.Tuple):
        return tuple(_official_action(subspace, step) for subspace in space)
    if isinstance(space, spaces.Discrete):
        n = int(space.n)
        if n == 1:
            return 0
        # Match the existing multi-step render gate; lane-change dynamics are
        # covered by highway_align_test until the official renderer is shared.
        return [1, min(n - 1, 3), min(n - 1, 3), 4][step % 4]
    if isinstance(space, spaces.Box):
        action = np.zeros(space.shape, dtype=space.dtype)
        if action.size:
            flat = action.reshape(-1)
            flat[0] = np.asarray([0.0, 0.25, -0.25, 0.0], dtype=space.dtype)[
                step % 4
            ]
            if flat.size > 1:
                flat[1] = np.asarray(
                    [0.0, -0.15, 0.15, 0.0], dtype=space.dtype
                )[step % 4]
        return action
    raise TypeError(f"Unsupported official highway action space: {space!r}")


def _envpool_action(action: Any) -> np.ndarray:
    if isinstance(action, tuple):
        return np.asarray(action)
    action_array = np.asarray(action)
    if action_array.shape:
        return action_array[None, ...]
    return action_array.reshape(1)


def _envpool_step(env: Any, case: _Case, action: Any) -> _Step:
    envpool_action = _envpool_action(action)
    if case.player_count == 1:
        obs, reward, terminated, truncated, info = env.step(envpool_action)
    else:
        obs, reward, terminated, truncated, info = env.step(
            envpool_action.reshape(-1), np.asarray([0], dtype=np.int32)
        )
    return _Step(obs, reward, terminated, truncated, info)


def _envpool_reset_obs(obs: Any, case: _Case) -> Any:
    return _envpool_obs(obs, case)


def _envpool_obs(obs: Any, case: _Case) -> Any:
    if isinstance(obs, dict):
        return {key: _envpool_obs(value, case) for key, value in obs.items()}
    obs_array = np.asarray(obs)
    if case.player_count == 1:
        return obs_array[0]
    return tuple(obs_array[player] for player in range(case.player_count))


def _envpool_scalar(value: Any) -> Any:
    value_array = np.asarray(value)
    assert value_array.shape[:1] == (1,), value_array.shape
    return value_array[0]


def _envpool_players(value: Any, player_count: int) -> tuple[Any, ...]:
    value_array = np.asarray(value)
    assert value_array.shape[:1] == (player_count,), value_array.shape
    return tuple(value_array[player] for player in range(player_count))


def _envpool_info(info: dict[str, Any]) -> dict[str, Any]:
    exposed_info: dict[str, Any] = {}
    for key in ("speed", "crashed", "is_success"):
        if key in info:
            exposed_info[key] = _envpool_scalar(info[key])
    return exposed_info


def _official_info(info: dict[str, Any]) -> dict[str, Any]:
    exposed = {
        key: info[key]
        for key in ("speed", "crashed", "is_success")
        if key in info
    }
    if "speed" in exposed:
        exposed["speed"] = np.float32(exposed["speed"])
    return exposed


def _assert_tree_bitwise(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_bitwise(actual[key], expected[key])
        return
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_tree_bitwise(actual_item, expected_item)
        return
    np.testing.assert_array_equal(actual, expected)


def _render_rgb(env: Any) -> np.ndarray:
    frame = env.render()
    assert frame is not None
    return np.asarray(frame, dtype=np.uint8)


def _patch_oracle_from_envpool(oracle: gym.Env, envpool_env: Any) -> None:
    _patch_oracle(oracle, _debug_state(envpool_env))


_RENDER_MISMATCH_PIXEL_LIMIT: dict[str, int] = {
    "highway-v0": 0,
    "highway-fast-v0": 0,
    "merge-v0": 500,
    "two-way-v0": 300,
    "u-turn-v0": 300,
    "exit-v0": 0,
    # These paths use pygame.transform.rotate / curved-lane rasterization
    # upstream.  Keep the residual to sprite/line edge pixels only.
    "intersection-v0": 1000,
    "intersection-v1": 1000,
    "intersection-multi-agent-v0": 1200,
    "intersection-multi-agent-v1": 1200,
    "lane-keeping-v0": 600,
    "parking-v0": 1600,
    "parking-ActionRepeat-v0": 1600,
    "parking-parked-v0": 1800,
    "racetrack-v0": 800,
    "racetrack-oval-v0": 800,
    "racetrack-large-v0": 800,
    "roundabout-v0": 1200,
}


def _assert_render_aligned(
    case: _Case, envpool_env: Any, oracle: gym.Env, step_label: str
) -> None:
    actual = _render_rgb(envpool_env)[0]
    expected = _render_rgb(oracle)
    assert actual.shape == expected.shape, (
        case.official_id,
        step_label,
        actual.shape,
        expected.shape,
    )
    mismatch_mask = np.any(actual != expected, axis=2)
    mismatch_pixels = int(mismatch_mask.sum())
    if mismatch_pixels == 0:
        return
    mismatch_channels = int(np.count_nonzero(actual != expected))
    limit = _RENDER_MISMATCH_PIXEL_LIMIT[case.official_id]
    assert mismatch_pixels <= limit, (
        f"{case.official_id} render mismatch at {step_label}: "
        f"{mismatch_pixels} pixels / {mismatch_channels} channel values; "
        f"limit={limit}"
    )


def _restore_official_idm_defaults() -> None:
    from highway_env.vehicle.behavior import IDMVehicle

    # IntersectionEnv mutates the global IDMVehicle class during reset.
    IDMVehicle.COMFORT_ACC_MAX = 3.0
    IDMVehicle.COMFORT_ACC_MIN = -5.0
    IDMVehicle.DISTANCE_WANTED = 10.0


class _HighwayOfficialAlignTest(absltest.TestCase):
    def test_reset_step_and_render_match_official_bitwise(self) -> None:
        patched_state_cases = [
            case
            for case in _CASES
            if case.official_id in ("highway-v0", "highway-fast-v0")
        ]
        for case in patched_state_cases:
            with self.subTest(official_id=case.official_id):
                env = make_gymnasium(
                    case.official_id,
                    num_envs=1,
                    seed=0,
                    render_mode="rgb_array",
                    vehicles_count=0,
                    initial_lane_id=1,
                )
                oracle = _make_oracle(case.official_id)
                try:
                    env_obs, _ = env.reset()
                    oracle.reset(seed=123)
                    _restore_official_idm_defaults()
                    _patch_oracle_from_envpool(oracle, env)
                    oracle_obs = oracle.unwrapped.observation_type.observe()
                    _assert_tree_bitwise(
                        _envpool_reset_obs(env_obs, case), oracle_obs
                    )
                    _assert_render_aligned(case, env, oracle, "reset")

                    for step in range(_OFFICIAL_ALIGN_STEPS):
                        action = _official_action(oracle.action_space, step)
                        actual = _envpool_step(env, case, action)
                        _patch_oracle_from_envpool(oracle, env)
                        expected = _Step(
                            obs=oracle.unwrapped.observation_type.observe(),
                            reward=oracle.unwrapped._reward(action),
                            terminated=oracle.unwrapped._is_terminated(),
                            truncated=oracle.unwrapped._is_truncated(),
                            info=oracle.unwrapped._info(
                                oracle.unwrapped.observation_type.observe(),
                                action,
                            ),
                        )

                        _assert_tree_bitwise(
                            _envpool_obs(actual.obs, case), expected.obs
                        )
                        np.testing.assert_array_equal(
                            _envpool_scalar(actual.reward),
                            np.float32(expected.reward),
                        )
                        np.testing.assert_array_equal(
                            _envpool_scalar(actual.terminated),
                            expected.terminated,
                        )
                        np.testing.assert_array_equal(
                            _envpool_scalar(actual.truncated),
                            expected.truncated,
                        )
                        _assert_tree_bitwise(
                            _envpool_info(actual.info),
                            _official_info(expected.info),
                        )
                        _assert_render_aligned(
                            case, env, oracle, f"step {step}"
                        )
                        if bool(expected.terminated) or bool(
                            expected.truncated
                        ):
                            break
                finally:
                    env.close()
                    oracle.close()

    def test_native_official_backend_steps_match_official_from_patched_state(
        self,
    ) -> None:
        for case in (
            _Case("merge-v0"),
            _Case("roundabout-v0"),
            _Case("two-way-v0"),
            _Case("u-turn-v0"),
        ):
            with self.subTest(official_id=case.official_id):
                env = make_gymnasium(
                    case.official_id,
                    num_envs=1,
                    seed=7,
                    render_mode="rgb_array",
                )
                oracle = _make_oracle(case.official_id)
                try:
                    env_obs, _ = env.reset()
                    oracle.reset(seed=123)
                    _restore_official_idm_defaults()
                    _patch_oracle_from_envpool(oracle, env)
                    _assert_tree_bitwise(
                        _envpool_reset_obs(env_obs, case),
                        oracle.unwrapped.observation_type.observe(),
                    )
                    _assert_render_aligned(case, env, oracle, "reset")

                    actions = (1, 3, 0, 2, 4)
                    for step in range(_OFFICIAL_BACKEND_ALIGN_STEPS):
                        action = actions[step % len(actions)]
                        _patch_oracle_from_envpool(oracle, env)
                        expected = _Step(*oracle.step(action))
                        actual = _envpool_step(env, case, action)

                        _assert_tree_bitwise(
                            _envpool_obs(actual.obs, case), expected.obs
                        )
                        np.testing.assert_array_equal(
                            _envpool_scalar(actual.reward),
                            np.float32(expected.reward),
                        )
                        np.testing.assert_array_equal(
                            _envpool_scalar(actual.terminated),
                            expected.terminated,
                        )
                        np.testing.assert_array_equal(
                            _envpool_scalar(actual.truncated),
                            expected.truncated,
                        )
                        _assert_tree_bitwise(
                            _envpool_info(actual.info),
                            _official_info(expected.info),
                        )
                        _assert_render_aligned(
                            case, env, oracle, f"step {step}"
                        )
                        if bool(expected.terminated) or bool(
                            expected.truncated
                        ):
                            break
                finally:
                    env.close()
                    oracle.close()

    def test_native_parking_steps_match_official_from_patched_state(
        self,
    ) -> None:
        for case in (
            _Case("parking-v0"),
            _Case("parking-ActionRepeat-v0"),
            _Case("parking-parked-v0"),
        ):
            with self.subTest(official_id=case.official_id):
                env = make_gymnasium(
                    case.official_id,
                    num_envs=1,
                    seed=11,
                    render_mode="rgb_array",
                )
                oracle = _make_oracle(case.official_id)
                try:
                    env_obs, _ = env.reset()
                    oracle.reset(seed=123)
                    _restore_official_idm_defaults()
                    _patch_oracle_from_envpool(oracle, env)
                    _assert_tree_bitwise(
                        _envpool_reset_obs(env_obs, case),
                        oracle.unwrapped.observation_type.observe(),
                    )
                    _assert_render_aligned(case, env, oracle, "reset")

                    for step in range(_OFFICIAL_ALIGN_STEPS):
                        action = _official_action(oracle.action_space, step)
                        actual = _envpool_step(env, case, action)
                        _patch_oracle_from_envpool(oracle, env)
                        oracle_obs = oracle.unwrapped.observation_type.observe()
                        expected = _Step(
                            obs=oracle_obs,
                            reward=oracle.unwrapped._reward(action),
                            terminated=oracle.unwrapped._is_terminated(),
                            truncated=oracle.unwrapped._is_truncated(),
                            info=oracle.unwrapped._info(oracle_obs, action),
                        )

                        _assert_tree_bitwise(
                            _envpool_obs(actual.obs, case), expected.obs
                        )
                        np.testing.assert_array_equal(
                            _envpool_scalar(actual.reward),
                            np.float32(expected.reward),
                        )
                        np.testing.assert_array_equal(
                            _envpool_scalar(actual.terminated),
                            expected.terminated,
                        )
                        np.testing.assert_array_equal(
                            _envpool_scalar(actual.truncated),
                            expected.truncated,
                        )
                        _assert_tree_bitwise(
                            _envpool_info(actual.info),
                            _official_info(expected.info),
                        )
                        _assert_render_aligned(
                            case, env, oracle, f"step {step}"
                        )
                        if bool(expected.terminated) or bool(
                            expected.truncated
                        ):
                            break
                finally:
                    env.close()
                    oracle.close()

    def test_native_exit_matches_official_from_patched_state(self) -> None:
        case = _Case("exit-v0")
        env = make_gymnasium(
            case.official_id, num_envs=1, seed=13, render_mode="rgb_array"
        )
        oracle = _make_oracle(case.official_id)
        try:
            env_obs, _ = env.reset()
            oracle.reset(seed=123)
            _restore_official_idm_defaults()
            _patch_oracle_from_envpool(oracle, env)
            _assert_tree_bitwise(
                _envpool_reset_obs(env_obs, case),
                oracle.unwrapped.observation_type.observe(),
            )
            _assert_render_aligned(case, env, oracle, "reset")

            for step in range(_EXIT_ALIGN_STEPS):
                action = _official_action(oracle.action_space, step)
                actual = _envpool_step(env, case, action)
                _patch_oracle_from_envpool(oracle, env)
                oracle_obs = oracle.unwrapped.observation_type.observe()
                expected = _Step(
                    obs=oracle_obs,
                    reward=oracle.unwrapped._reward(action),
                    terminated=oracle.unwrapped._is_terminated(),
                    truncated=oracle.unwrapped._is_truncated(),
                    info=oracle.unwrapped._info(oracle_obs, action),
                )
                _assert_tree_bitwise(
                    _envpool_obs(actual.obs, case), expected.obs
                )
                np.testing.assert_array_equal(
                    _envpool_scalar(actual.reward), np.float32(expected.reward)
                )
                np.testing.assert_array_equal(
                    _envpool_scalar(actual.terminated), expected.terminated
                )
                np.testing.assert_array_equal(
                    _envpool_scalar(actual.truncated), expected.truncated
                )
                _assert_tree_bitwise(
                    _envpool_info(actual.info), _official_info(expected.info)
                )
                _assert_render_aligned(case, env, oracle, f"step {step}")
                if bool(expected.terminated) or bool(expected.truncated):
                    break
        finally:
            env.close()
            oracle.close()

    def test_native_intersection_matches_official_from_patched_state(
        self,
    ) -> None:
        for case in (_Case("intersection-v0"), _Case("intersection-v1")):
            with self.subTest(official_id=case.official_id):
                env = make_gymnasium(
                    case.official_id,
                    num_envs=1,
                    seed=17,
                    render_mode="rgb_array",
                )
                oracle = _make_oracle(case.official_id)
                try:
                    env_obs, _ = env.reset()
                    oracle.reset(seed=123)
                    _restore_official_idm_defaults()
                    _patch_oracle_from_envpool(oracle, env)
                    _assert_tree_bitwise(
                        _envpool_reset_obs(env_obs, case),
                        oracle.unwrapped.observation_type.observe(),
                    )
                    _assert_render_aligned(case, env, oracle, "reset")

                    for step in range(_INTERSECTION_ALIGN_STEPS):
                        action = (
                            [1, 2, 0, 1, 2, 0][step % 6]
                            if isinstance(oracle.action_space, spaces.Discrete)
                            else _official_action(oracle.action_space, step)
                        )
                        actual = _envpool_step(env, case, action)
                        _patch_oracle_from_envpool(oracle, env)
                        oracle_obs = oracle.unwrapped.observation_type.observe()
                        expected = _Step(
                            obs=oracle_obs,
                            reward=oracle.unwrapped._reward(action),
                            terminated=oracle.unwrapped._is_terminated(),
                            truncated=oracle.unwrapped._is_truncated(),
                            info=oracle.unwrapped._info(oracle_obs, action),
                        )
                        _assert_tree_bitwise(
                            _envpool_obs(actual.obs, case), expected.obs
                        )
                        np.testing.assert_array_equal(
                            _envpool_scalar(actual.reward),
                            np.float32(expected.reward),
                        )
                        np.testing.assert_array_equal(
                            _envpool_scalar(actual.terminated),
                            expected.terminated,
                        )
                        np.testing.assert_array_equal(
                            _envpool_scalar(actual.truncated),
                            expected.truncated,
                        )
                        _assert_tree_bitwise(
                            _envpool_info(actual.info),
                            _official_info(expected.info),
                        )
                        _assert_render_aligned(
                            case, env, oracle, f"step {step}"
                        )
                        if bool(expected.terminated) or bool(
                            expected.truncated
                        ):
                            break
                finally:
                    env.close()
                    oracle.close()

    def test_native_multi_agent_intersection_matches_official_patched_state(
        self,
    ) -> None:
        for case in (
            _Case("intersection-multi-agent-v0", player_count=2),
            _Case("intersection-multi-agent-v1", player_count=2),
        ):
            with self.subTest(official_id=case.official_id):
                env = make_gymnasium(
                    case.official_id,
                    num_envs=1,
                    seed=19,
                    render_mode="rgb_array",
                )
                oracle = _make_oracle(case.official_id)
                try:
                    env_obs, _ = env.reset()
                    oracle.reset(seed=123)
                    _restore_official_idm_defaults()
                    _patch_oracle_from_envpool(oracle, env)
                    _assert_tree_bitwise(
                        _envpool_reset_obs(env_obs, case),
                        oracle.unwrapped.observation_type.observe(),
                    )
                    _assert_render_aligned(case, env, oracle, "reset")

                    actions = (
                        (1, 1),
                        (2, 1),
                        (1, 0),
                        (0, 2),
                        (1, 1),
                    )
                    for step in range(_OFFICIAL_ALIGN_STEPS):
                        action = actions[step % len(actions)]
                        actual = _envpool_step(env, case, action)
                        _patch_oracle_from_envpool(oracle, env)
                        oracle_obs = oracle.unwrapped.observation_type.observe()
                        expected_reward = tuple(
                            np.float32(
                                oracle.unwrapped._agent_reward(action, vehicle)
                            )
                            for vehicle in oracle.unwrapped.controlled_vehicles
                        )
                        expected_terminated = tuple(
                            oracle.unwrapped._agent_is_terminal(vehicle)
                            for vehicle in oracle.unwrapped.controlled_vehicles
                        )

                        _assert_tree_bitwise(
                            _envpool_obs(actual.obs, case), oracle_obs
                        )
                        _assert_tree_bitwise(
                            _envpool_players(actual.reward, case.player_count),
                            expected_reward,
                        )
                        np.testing.assert_array_equal(
                            _envpool_scalar(actual.terminated),
                            any(expected_terminated),
                        )
                        np.testing.assert_array_equal(
                            _envpool_scalar(actual.truncated),
                            oracle.unwrapped._is_truncated(),
                        )
                        _assert_render_aligned(
                            case, env, oracle, f"step {step}"
                        )
                        if any(expected_terminated) or bool(
                            _envpool_scalar(actual.truncated)
                        ):
                            break
                finally:
                    env.close()
                    oracle.close()

    def test_native_lane_keeping_matches_official_without_observation_noise(
        self,
    ) -> None:
        case = _Case("lane-keeping-v0")
        env = make_gymnasium(
            case.official_id, num_envs=1, seed=23, render_mode="rgb_array"
        )
        oracle = _make_oracle(
            case.official_id,
            {"state_noise": 0.0, "derivative_noise": 0.0},
        )
        try:
            env_obs, _ = env.reset()
            oracle.reset(seed=123)
            _restore_official_idm_defaults()
            _patch_oracle_from_envpool(oracle, env)
            oracle_obs = oracle.unwrapped.observation_type.observe()
            _assert_tree_bitwise(_envpool_reset_obs(env_obs, case), oracle_obs)
            _assert_render_aligned(case, env, oracle, "reset")

            actions = (
                np.asarray([0.0], dtype=np.float32),
                np.asarray([0.25], dtype=np.float32),
            )
            for step in range(_LANE_KEEPING_ALIGN_STEPS):
                action = actions[step % len(actions)]
                expected = _Step(*oracle.step(action))
                actual = _envpool_step(env, case, action)
                _assert_tree_bitwise(
                    _envpool_obs(actual.obs, case), expected.obs
                )
                np.testing.assert_array_equal(
                    _envpool_scalar(actual.reward), np.float32(expected.reward)
                )
                np.testing.assert_array_equal(
                    _envpool_scalar(actual.terminated), expected.terminated
                )
                np.testing.assert_array_equal(
                    _envpool_scalar(actual.truncated), expected.truncated
                )
                _assert_render_aligned(case, env, oracle, f"step {step}")
                if bool(expected.terminated) or bool(expected.truncated):
                    break
        finally:
            env.close()
            oracle.close()

    def test_native_racetrack_matches_official_from_patched_state(self) -> None:
        actions = (
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0.25], dtype=np.float32),
            np.asarray([-0.25], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0.125], dtype=np.float32),
        )
        for case in (
            _Case("racetrack-v0"),
            _Case("racetrack-large-v0"),
            _Case("racetrack-oval-v0"),
        ):
            with self.subTest(official_id=case.official_id):
                env = make_gymnasium(
                    case.official_id,
                    num_envs=1,
                    seed=29,
                    render_mode="rgb_array",
                )
                oracle = _make_oracle(case.official_id)
                try:
                    env_obs, _ = env.reset()
                    oracle.reset(seed=123)
                    _patch_oracle_from_envpool(oracle, env)
                    _assert_tree_bitwise(
                        _envpool_reset_obs(env_obs, case),
                        oracle.unwrapped.observation_type.observe(),
                    )
                    _assert_render_aligned(case, env, oracle, "reset")

                    for step in range(_RACETRACK_ALIGN_STEPS):
                        action = actions[step % len(actions)]
                        expected = _Step(*oracle.step(action))
                        actual = _envpool_step(env, case, action)
                        _assert_tree_bitwise(
                            _envpool_obs(actual.obs, case), expected.obs
                        )
                        np.testing.assert_array_equal(
                            _envpool_scalar(actual.reward),
                            np.float32(expected.reward),
                        )
                        np.testing.assert_array_equal(
                            _envpool_scalar(actual.terminated),
                            expected.terminated,
                        )
                        np.testing.assert_array_equal(
                            _envpool_scalar(actual.truncated),
                            expected.truncated,
                        )
                        _assert_render_aligned(
                            case, env, oracle, f"step {step}"
                        )
                        if bool(expected.terminated) or bool(
                            expected.truncated
                        ):
                            break
                finally:
                    env.close()
                    oracle.close()


if __name__ == "__main__":
    absltest.main()
