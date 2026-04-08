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
"""Alignment tests for the C++ Highway backend."""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from absl.testing import absltest
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle

import envpool.highway.registration  # noqa: F401
from envpool.registration import make_gymnasium

_ALIGN_ACTIONS = (1, 3, 3, 2, 1, 1, 0, 4, 4, 1, 2, 1)
_DEFAULT_ALIGN_CONFIG = {
    "vehicles_count": 0,
    "lanes_count": 3,
    "initial_lane_id": 1,
    "duration": 40,
    "simulation_frequency": 15,
    "policy_frequency": 1,
}

_STRAIGHT_ROAD_ALIGN_CONFIGS = (
    ("highway_v0_no_traffic", "Highway-v0", "highway-v0", {}),
    (
        "highway_v0_two_lane_left_ego",
        "Highway-v0",
        "highway-v0",
        {"lanes_count": 2, "initial_lane_id": 0},
    ),
    (
        "highway_v0_two_hz_policy",
        "Highway-v0",
        "highway-v0",
        {"lanes_count": 4, "initial_lane_id": 2, "policy_frequency": 2},
    ),
    (
        "highway_fast_v0_no_traffic",
        "HighwayFast-v0",
        "highway-fast-v0",
        {
            "duration": 30,
            "simulation_frequency": 5,
            "lanes_count": 3,
            "initial_lane_id": 1,
        },
    ),
)


def _debug_state(env: Any) -> Any:
    return cast(Any, env).debug_states([0])[0]


def _straight_road_config(**overrides: Any) -> dict[str, Any]:
    config = dict(_DEFAULT_ALIGN_CONFIG)
    config.update(overrides)
    return config


def _assert_scalar_matches_float32(
    test_case: absltest.TestCase, actual: Any, expected: float
) -> None:
    test_case.assertEqual(np.asarray(actual).dtype, np.dtype(np.float32))
    np.testing.assert_array_equal(actual, np.float32(expected))


def _make_oracle(oracle_env_id: str, config: dict[str, Any]) -> gym.Env:
    oracle_config: dict[str, Any] = {
        "lanes_count": config["lanes_count"],
        "vehicles_count": config["vehicles_count"],
        "duration": config["duration"],
        "simulation_frequency": config["simulation_frequency"],
        "policy_frequency": config["policy_frequency"],
        "offscreen_rendering": True,
    }
    if config["initial_lane_id"] >= 0:
        oracle_config["initial_lane_id"] = config["initial_lane_id"]
    return gym.make(
        oracle_env_id, render_mode="rgb_array", config=oracle_config
    )


def _patch_oracle(oracle: gym.Env, debug_state: Any) -> None:
    env = cast(Any, oracle.unwrapped)
    road = env.road
    vehicles = []
    for i, source in enumerate(debug_state.vehicles):
        lane_index = ("0", "1", int(source.lane_index))
        target_lane_index = ("0", "1", int(source.target_lane_index))
        if int(source.kind) == 0:
            vehicle = MDPVehicle(
                road,
                np.asarray([source.x, source.y], dtype=np.float64),
                heading=float(source.heading),
                speed=float(source.speed),
                target_lane_index=target_lane_index,
                target_speed=float(source.target_speed),
                target_speeds=np.asarray(
                    [
                        source.target_speed0,
                        source.target_speed1,
                        source.target_speed2,
                    ],
                    dtype=np.float64,
                ),
            )
            vehicle.speed_index = int(source.speed_index)
            vehicle.target_speed = float(source.target_speed)
            if i == 0:
                env.controlled_vehicles = [vehicle]
        else:
            vehicle = IDMVehicle(
                road,
                np.asarray([source.x, source.y], dtype=np.float64),
                heading=float(source.heading),
                speed=float(source.speed),
                target_lane_index=target_lane_index,
                target_speed=float(source.target_speed),
                timer=float(source.timer),
            )
            vehicle.DELTA = float(source.idm_delta)
        vehicle.lane_index = lane_index
        vehicle.lane = road.network.get_lane(lane_index)
        vehicle.target_lane_index = target_lane_index
        vehicle.crashed = bool(source.crashed)
        vehicle.check_collisions = bool(source.check_collisions)
        vehicles.append(vehicle)
    road.vehicles = vehicles
    env.time = float(debug_state.time)
    frames = int(
        debug_state.simulation_frequency // debug_state.policy_frequency
    )
    env.steps = int(debug_state.elapsed_step) * frames
    env.done = bool(vehicles[0].crashed) if vehicles else False


class _HighwayAlignTest(absltest.TestCase):
    def test_space_matches_upstream(self) -> None:
        oracle = gym.make("highway-v0")
        env = make_gymnasium("Highway-v0")
        try:
            self.assertEqual(
                env.observation_space.shape, oracle.observation_space.shape
            )
            self.assertEqual(env.action_space.n, oracle.action_space.n)
        finally:
            oracle.close()
            env.close()

    def assert_controlled_vehicle_aligns_with_upstream(
        self,
        task_id: str,
        oracle_env_id: str,
        config: dict[str, Any],
    ) -> None:
        env = make_gymnasium(task_id, num_envs=1, seed=7, **config)
        oracle = _make_oracle(oracle_env_id, config)
        try:
            obs, _ = env.reset()
            oracle.reset(seed=123)
            _patch_oracle(oracle, _debug_state(env))
            np.testing.assert_array_equal(
                obs[0],
                oracle.unwrapped.observation_type.observe(),
            )

            for action in _ALIGN_ACTIONS:
                (
                    oracle_obs,
                    oracle_rew,
                    oracle_term,
                    oracle_trunc,
                    oracle_info,
                ) = oracle.step(action)
                obs, rew, term, trunc, info = env.step(
                    np.asarray([action], dtype=np.int64)
                )
                np.testing.assert_array_equal(obs[0], oracle_obs)
                _assert_scalar_matches_float32(self, rew[0], oracle_rew)
                self.assertEqual(bool(term[0]), oracle_term)
                self.assertEqual(bool(trunc[0]), oracle_trunc)
                _assert_scalar_matches_float32(
                    self, info["speed"][0], oracle_info["speed"]
                )
                self.assertEqual(
                    bool(info["crashed"][0]), oracle_info["crashed"]
                )

                _patch_oracle(oracle, _debug_state(env))
        finally:
            env.close()
            oracle.close()

    def test_controlled_vehicle_aligns_with_upstream_configs(
        self,
    ) -> None:
        for (
            name,
            task_id,
            oracle_env_id,
            overrides,
        ) in _STRAIGHT_ROAD_ALIGN_CONFIGS:
            with self.subTest(name=name):
                self.assert_controlled_vehicle_aligns_with_upstream(
                    task_id,
                    oracle_env_id,
                    _straight_road_config(**overrides),
                )


if __name__ == "__main__":
    absltest.main()
