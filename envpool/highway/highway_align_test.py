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
import numpy as np
from absl.testing import absltest

from envpool.highway.highway_oracle_util import (
    prepare_official_oracle_import,
    register_highway_envs,
)
from envpool.registration import make_gymnasium

register_highway_envs()
prepare_official_oracle_import()

_ALIGN_ACTIONS = (1, 3, 3, 2, 1, 1, 0, 4, 4, 1, 2, 1)
_DEFAULT_ALIGN_CONFIG = {
    "vehicles_count": 0,
    "lanes_count": 3,
    "initial_lane_id": 1,
    "duration": 40,
    "simulation_frequency": 15,
    "policy_frequency": 1,
}

_STRAIGHT_ROAD_ALIGN_CONFIGS: tuple[
    tuple[str, str, str, dict[str, Any]], ...
] = (
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
    import highway_env  # noqa: F401

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


def _patch_oracle_road(oracle: gym.Env, debug_state: Any) -> None:
    from highway_env.road.lane import (
        CircularLane,
        LineType,
        SineLane,
        StraightLane,
    )
    from highway_env.road.road import RoadNetwork
    from highway_env.vehicle.objects import Landmark, Obstacle

    env = cast(Any, oracle.unwrapped)
    road_lanes = list(getattr(debug_state, "road_lanes", ()))
    if road_lanes:
        line_types = (
            LineType.NONE,
            LineType.STRIPED,
            LineType.CONTINUOUS,
            LineType.CONTINUOUS_LINE,
        )
        network = RoadNetwork()
        for source in road_lanes:
            line_pair = (
                line_types[int(source.line_type0)],
                line_types[int(source.line_type1)],
            )
            common = {
                "width": float(source.width),
                "line_types": line_pair,
                "forbidden": bool(source.forbidden),
                "speed_limit": float(source.speed_limit),
                "priority": int(source.priority),
            }
            if int(source.kind) == 2:
                lane = CircularLane(
                    np.asarray(
                        [source.center_x, source.center_y],
                        dtype=np.float64,
                    ),
                    float(source.radius),
                    float(source.start_phase),
                    float(source.end_phase),
                    clockwise=bool(source.clockwise),
                    **common,
                )
            elif int(source.kind) == 1:
                lane = SineLane(
                    np.asarray([source.start_x, source.start_y], dtype=np.float64),
                    np.asarray([source.end_x, source.end_y], dtype=np.float64),
                    float(source.amplitude),
                    float(source.pulsation),
                    float(source.phase),
                    **common,
                )
            else:
                lane = StraightLane(
                    np.asarray([source.start_x, source.start_y], dtype=np.float64),
                    np.asarray([source.end_x, source.end_y], dtype=np.float64),
                    **common,
                )
            network.add_lane(str(getattr(source, "from")), str(source.to), lane)
        env.road.network = network

    road_objects = list(getattr(debug_state, "road_objects", ()))
    if road_objects:
        objects = []
        for source in road_objects:
            object_cls = Landmark if int(source.kind) == 1 else Obstacle
            object_ = object_cls(
                env.road,
                np.asarray([source.x, source.y], dtype=np.float64),
                heading=float(source.heading),
                speed=float(source.speed),
            )
            object_.LENGTH = float(source.length)
            object_.WIDTH = float(source.width)
            object_.collidable = bool(source.collidable)
            object_.solid = bool(source.solid)
            object_.check_collisions = bool(source.check_collisions)
            object_.crashed = bool(source.crashed)
            object_.hit = bool(source.hit)
            object_.lane_index = env.road.network.get_closest_lane_index(
                object_.position, object_.heading
            )
            object_.lane = env.road.network.get_lane(object_.lane_index)
            objects.append(object_)
        env.road.objects = objects


def _patch_oracle(oracle: gym.Env, debug_state: Any) -> None:
    from highway_env.vehicle.behavior import IDMVehicle
    from highway_env.vehicle.controller import MDPVehicle
    from highway_env.vehicle.kinematics import Vehicle
    from highway_env.vehicle.objects import Landmark

    env = cast(Any, oracle.unwrapped)
    _patch_oracle_road(oracle, debug_state)
    road = env.road
    vehicles = []
    controlled_vehicles = []
    landmarks = []
    is_parking = str(getattr(debug_state, "scenario", "")).startswith("parking")
    is_lane_keeping = str(getattr(debug_state, "scenario", "")).startswith(
        "lane_keeping"
    )
    is_plain_continuous = str(getattr(debug_state, "scenario", "")).startswith((
        "racetrack",
        "lane_keeping",
    ))
    for i, source in enumerate(debug_state.vehicles):
        lane_index = (
            str(getattr(source, "lane_from", "0")),
            str(getattr(source, "lane_to", "1")),
            int(source.lane_index),
        )
        target_lane_index = (
            str(getattr(source, "target_lane_from", "0")),
            str(getattr(source, "target_lane_to", "1")),
            int(source.target_lane_index),
        )
        kind = int(source.kind)
        if is_lane_keeping and i == 0:
            vehicle = env.vehicle
            vehicle.road = road
            controlled_vehicles.append(vehicle)
        elif is_parking or is_plain_continuous:
            vehicle = Vehicle(
                road,
                np.asarray([source.x, source.y], dtype=np.float64),
                heading=float(source.heading),
                speed=float(source.speed),
            )
            if is_plain_continuous and i == 0:
                controlled_vehicles.append(vehicle)
        elif kind in (0, 2):
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
                env.vehicle = vehicle
            controlled_vehicles.append(vehicle)
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
            vehicle.enable_lane_change = bool(
                getattr(source, "enable_lane_change", True)
            )
            if i == 0:
                env.vehicle = vehicle
        vehicle.lane_index = lane_index
        vehicle.lane = road.network.get_lane(lane_index)
        vehicle.target_lane_index = target_lane_index
        vehicle.speed = float(source.speed)
        vehicle.heading = float(source.heading)
        vehicle.position = np.asarray([source.x, source.y], dtype=np.float64)
        vehicle.action = {"steering": 0.0, "acceleration": 0.0}
        if getattr(source, "route_from", []):
            vehicle.route = [
                (
                    str(route_from),
                    str(route_to),
                    None if int(route_id) < 0 else int(route_id),
                )
                for route_from, route_to, route_id in zip(
                    source.route_from,
                    source.route_to,
                    source.route_id,
                    strict=True,
                )
            ]
        vehicle.crashed = bool(source.crashed)
        vehicle.check_collisions = bool(source.check_collisions)
        if bool(getattr(source, "has_goal", False)):
            goal = Landmark(
                road,
                np.asarray([source.goal_x, source.goal_y], dtype=np.float64),
                heading=float(source.goal_heading),
                speed=float(source.goal_speed),
            )
            vehicle.goal = goal
            landmarks.append(goal)
            if i == 0:
                env.vehicle = vehicle
                env.controlled_vehicles = [vehicle]
        vehicles.append(vehicle)
    road.vehicles = vehicles
    if controlled_vehicles:
        env.vehicle = controlled_vehicles[0]
        env.controlled_vehicles = controlled_vehicles
        agents_observation_types = getattr(
            env.observation_type, "agents_observation_types", ()
        )
        for obs_type, vehicle in zip(
            agents_observation_types, controlled_vehicles, strict=False
        ):
            obs_type.observer_vehicle = vehicle
    if landmarks and not getattr(debug_state, "road_objects", ()):
        road.objects = [
            obj for obj in road.objects if obj.__class__.__name__ != "Landmark"
        ] + landmarks
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
            assert isinstance(env.action_space, gym.spaces.Discrete)
            assert isinstance(oracle.action_space, gym.spaces.Discrete)
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
                cast(Any, oracle.unwrapped).observation_type.observe(),
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
                _assert_scalar_matches_float32(self, rew[0], float(oracle_rew))
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
