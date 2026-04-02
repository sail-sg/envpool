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
"""Alignment tests for the C++ MiniGrid backend."""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from absl.testing import absltest
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj
from minigrid.minigrid_env import MiniGridEnv as UpstreamMiniGridEnv

import envpool.minigrid.registration  # noqa: F401
from envpool.minigrid import decode_mission
from envpool.registration import list_all_envs, make_gymnasium

_EMPTY = 1
_DOOR = 4
_BALL = 6
_BOX = 7
_RED = 0
_BLUE = 2

_TYPE_NAMES = {
    0: "unseen",
    1: "empty",
    2: "wall",
    3: "floor",
    4: "door",
    5: "key",
    6: "ball",
    7: "box",
    8: "goal",
    9: "lava",
    10: "agent",
}
_COLOR_NAMES = {
    0: "red",
    1: "green",
    2: "blue",
    3: "purple",
    4: "yellow",
    5: "grey",
}


def _mission_from_obs(obs: dict[str, np.ndarray]) -> str:
    mission = obs["mission"]
    arr = np.asarray(mission)
    if arr.ndim == 1:
        return cast(str, decode_mission(arr))
    decoded = cast(np.ndarray, decode_mission(arr))
    return cast(str, decoded[0])


def _debug_state(env: Any) -> Any:
    debug_states = cast(Any, env)._debug_states(np.asarray([0], dtype=np.int32))
    return debug_states[0]


def _decode_obj(type_idx: int, color_idx: int, state: int) -> WorldObj | None:
    return WorldObj.decode(int(type_idx), int(color_idx), int(state))


def _obj_at(grid: Grid, pos: tuple[int, int]) -> WorldObj | None:
    if pos[0] < 0 or pos[1] < 0:
        return None
    return grid.get(*pos)


def _matching_carrying_obj(
    env: Any, target_type: int, target_color: int
) -> WorldObj | None:
    carrying = cast(WorldObj | None, env.carrying)
    if carrying is None:
        return None
    if carrying.type != _TYPE_NAMES[target_type]:
        return None
    if carrying.color != _COLOR_NAMES[target_color]:
        return None
    return carrying


def _rebuild_grid(debug_state: Any) -> Grid:
    width = int(debug_state.width)
    height = int(debug_state.height)
    grid = Grid(width, height)
    grid_data = np.asarray(debug_state.grid, dtype=np.uint8).reshape((
        width,
        height,
        3,
    ))
    contains_data = np.asarray(
        debug_state.grid_contains, dtype=np.uint8
    ).reshape((width, height, 3))
    for x in range(width):
        for y in range(height):
            obj = _decode_obj(*grid_data[x, y])
            if obj is not None:
                contains = _decode_obj(*contains_data[x, y])
                if contains is not None:
                    obj.contains = contains
                obj.init_pos = (x, y)
                obj.cur_pos = (x, y)
            grid.set(x, y, obj)
    return grid


def _rebuild_carrying(debug_state: Any) -> WorldObj | None:
    if not debug_state.has_carrying:
        return None
    carrying = _decode_obj(
        int(debug_state.carrying_type),
        int(debug_state.carrying_color),
        int(debug_state.carrying_state),
    )
    assert carrying is not None
    if debug_state.carrying_has_contains:
        contains = _decode_obj(
            int(debug_state.carrying_contains_type),
            int(debug_state.carrying_contains_color),
            int(debug_state.carrying_contains_state),
        )
        if contains is not None:
            carrying.contains = contains
    carrying.init_pos = None
    carrying.cur_pos = np.array([-1, -1])
    return carrying


def _patch_env_state(env: Any, debug_state: Any, elapsed_step: int) -> None:
    env.grid = _rebuild_grid(debug_state)
    env.agent_pos = tuple(debug_state.agent_pos)
    env.agent_dir = int(debug_state.agent_dir)
    env.mission = debug_state.mission
    env.carrying = _rebuild_carrying(debug_state)
    env.step_count = int(elapsed_step)

    env_name = debug_state.env_name
    if env_name == "dynamic_obstacles":
        obstacles = []
        obstacle_positions = list(debug_state.obstacle_positions)
        for i in range(0, len(obstacle_positions), 2):
            obj = _obj_at(
                env.grid,
                (int(obstacle_positions[i]), int(obstacle_positions[i + 1])),
            )
            assert obj is not None
            obstacles.append(obj)
        env.obstacles = obstacles
    elif env_name == "fetch":
        env.targetType = _TYPE_NAMES[int(debug_state.target_type)]
        env.targetColor = _COLOR_NAMES[int(debug_state.target_color)]
    elif env_name == "goto_door":
        env.target_pos = tuple(debug_state.target_pos)
        env.target_color = _COLOR_NAMES[int(debug_state.target_color)]
    elif env_name == "goto_object":
        env.targetType = _TYPE_NAMES[int(debug_state.target_type)]
        env.target_pos = tuple(debug_state.target_pos)
        env.target_color = _COLOR_NAMES[int(debug_state.target_color)]
    elif env_name == "put_near":
        env.move_type = _TYPE_NAMES[int(debug_state.move_type)]
        env.moveColor = _COLOR_NAMES[int(debug_state.move_color)]
        env.move_pos = tuple(debug_state.move_pos)
        env.target_type = _TYPE_NAMES[int(debug_state.target_type)]
        env.target_color = _COLOR_NAMES[int(debug_state.target_color)]
        env.target_pos = tuple(debug_state.target_pos)
    elif env_name == "red_blue_door":
        red_door = None
        blue_door = None
        for x in range(env.grid.width):
            for y in range(env.grid.height):
                obj = env.grid.get(x, y)
                if obj is None or obj.type != "door":
                    continue
                if obj.color == "red":
                    red_door = obj
                elif obj.color == "blue":
                    blue_door = obj
        env.red_door = red_door
        env.blue_door = blue_door
    elif env_name == "memory":
        env.success_pos = tuple(debug_state.success_pos)
        env.failure_pos = tuple(debug_state.failure_pos)
    elif env_name == "unlock":
        env.door = None
        for x in range(env.grid.width):
            for y in range(env.grid.height):
                obj = env.grid.get(x, y)
                if obj is not None and obj.type == "door":
                    env.door = obj
                    break
            if env.door is not None:
                break
    elif env_name in {
        "unlock_pickup",
        "blocked_unlock_pickup",
        "key_corridor",
        "obstructed_maze_1dlhb",
        "obstructed_maze_full",
        "obstructed_maze_full_v1",
    }:
        env.obj = _obj_at(env.grid, tuple(debug_state.target_pos))
        if env.obj is None:
            env.obj = _matching_carrying_obj(
                env,
                int(debug_state.target_type),
                int(debug_state.target_color),
            )


def _step_dynamic_obstacles(
    env: Any, pre_state: Any, post_state: Any, act: int
) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
    front_cell = env.grid.get(*env.front_pos)
    not_clear = front_cell is not None and front_cell.type != "goal"

    old_positions = list(pre_state.obstacle_positions)
    new_positions = list(post_state.obstacle_positions)
    assert len(old_positions) == len(new_positions)
    assert len(old_positions) == len(env.obstacles) * 2

    for idx, obstacle in enumerate(env.obstacles):
        old_pos = (int(old_positions[2 * idx]), int(old_positions[2 * idx + 1]))
        new_pos = (int(new_positions[2 * idx]), int(new_positions[2 * idx + 1]))
        if new_pos != old_pos:
            env.grid.set(old_pos[0], old_pos[1], None)
            env.grid.set(new_pos[0], new_pos[1], obstacle)
            obstacle.init_pos = new_pos
            obstacle.cur_pos = new_pos

    obs, reward, terminated, truncated, info = UpstreamMiniGridEnv.step(
        env, act
    )
    if act == env.actions.forward and not_clear:
        reward = -1
        terminated = True
    return obs, float(reward), bool(terminated), bool(truncated), info


class _MiniGridEnvPoolAlignTest(absltest.TestCase):
    def minigrid_task_ids(self) -> list[str]:
        task_ids = sorted(
            task_id
            for task_id in list_all_envs()
            if task_id.startswith("MiniGrid-")
        )
        self.assertLen(task_ids, 75)
        return task_ids

    def check_spec(
        self, spec0: gym.spaces.Space, spec1: gym.spaces.Space
    ) -> None:
        self.assertEqual(spec0.dtype, spec1.dtype)
        if isinstance(spec0, gym.spaces.Discrete):
            assert isinstance(spec1, gym.spaces.Discrete)
            self.assertEqual(spec0.n, spec1.n)
        elif isinstance(spec0, gym.spaces.Box):
            assert isinstance(spec1, gym.spaces.Box)
            np.testing.assert_allclose(spec0.low, spec1.low)
            np.testing.assert_allclose(spec0.high, spec1.high)

    def run_align_check(
        self,
        task_id: str,
        total: int = 100,
        **kwargs: Any,
    ) -> None:
        env0 = gym.make(task_id)
        env1 = make_gymnasium(task_id, num_envs=1, seed=0, **kwargs)
        obs_space0 = cast(Any, env0.observation_space)
        self.check_spec(
            obs_space0["direction"], env1.observation_space["direction"]
        )
        self.check_spec(obs_space0["image"], env1.observation_space["image"])
        self.check_spec(env0.action_space, env1.action_space)

        obs1_reset, info1_reset = env1.reset()
        obs0_reset, _ = env0.reset(seed=0)
        _patch_env_state(
            cast(Any, env0.unwrapped),
            _debug_state(env1),
            int(info1_reset["elapsed_step"][0]),
        )
        self.assertEqual(
            cast(Any, env0.unwrapped).mission, _mission_from_obs(obs1_reset)
        )

        done0 = False
        for _ in range(total):
            act = env0.action_space.sample()
            pre_state = _debug_state(env1)
            obs1, rew1, term1, trunc1, info1 = env1.step(np.array([act]))
            post_state = _debug_state(env1)
            if done0:
                env0.reset()
                _patch_env_state(
                    cast(Any, env0.unwrapped),
                    post_state,
                    int(info1["elapsed_step"][0]),
                )
                done0 = bool(term1[0] or trunc1[0])
                continue

            if pre_state.env_name == "dynamic_obstacles":
                obs0, rew0, term0, trunc0, _ = _step_dynamic_obstacles(
                    cast(Any, env0.unwrapped), pre_state, post_state, act
                )
            else:
                obs0, rew0, term0, trunc0, _ = cast(Any, env0.step(act))
            np.testing.assert_array_equal(
                obs0["direction"], obs1["direction"][0]
            )
            np.testing.assert_array_equal(obs0["image"], obs1["image"][0])
            self.assertEqual(obs0["mission"], _mission_from_obs(obs1))
            np.testing.assert_allclose(float(rew0), float(rew1[0]), rtol=1e-6)
            # EnvPool's gymnasium wrapper maps timeout steps to
            # `terminated=False, truncated=True` via `done & ~trunc`.
            self.assertEqual(bool(term0 and not trunc0), bool(term1[0]))
            self.assertEqual(bool(trunc0), bool(trunc1[0]))
            np.testing.assert_array_equal(
                np.asarray(cast(Any, env0.unwrapped).agent_pos),
                info1["agent_pos"][0],
            )
            _patch_env_state(
                cast(Any, env0.unwrapped),
                post_state,
                int(info1["elapsed_step"][0]),
            )
            done0 = bool(term0 or trunc0)

    def test_registered_minigrid_envs(self) -> None:
        for task_id in self.minigrid_task_ids():
            with self.subTest(task_id=task_id):
                print(f"align {task_id}", flush=True)
                self.run_align_check(task_id)


if __name__ == "__main__":
    absltest.main()
