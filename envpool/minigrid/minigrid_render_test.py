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
"""Render tests for MiniGrid environments."""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from absl.testing import absltest
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj
from minigrid.minigrid_env import MiniGridEnv as UpstreamMiniGridEnv

import envpool.minigrid.registration  # noqa: F401
from envpool.registration import list_all_envs, make_gymnasium

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


def _patch_wfc_oracle_reset() -> None:
    import minigrid.envs.wfc as wfc_module

    class _WFCOracleEnv(UpstreamMiniGridEnv):
        def __init__(
            self,
            wfc_config: Any = "MazeSimple",
            size: int = 25,
            ensure_connected: bool = True,
            max_steps: int | None = None,
            **kwargs: Any,
        ) -> None:
            del wfc_config, ensure_connected
            if max_steps is None:
                max_steps = size * 20
            super().__init__(
                mission_space=MissionSpace(mission_func=self._gen_mission),
                width=size,
                height=size,
                max_steps=max_steps,
                **kwargs,
            )

        @staticmethod
        def _gen_mission() -> str:
            return "traverse the maze to get to the goal"

        def _gen_grid(self, width: int, height: int) -> None:
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)
            self.grid.set(width - 2, height - 2, WorldObj.decode(8, 1, 0))
            self.agent_pos = (1, 1)
            self.agent_dir = 0
            self.mission = self._gen_mission()

    wfc_module.WFCEnv = _WFCOracleEnv


_patch_wfc_oracle_reset()
_REPRESENTATIVE_TASK_IDS = (
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-BlockedUnlockPickup-v0",
)
_RENDER_STEPS = 3
_TASK_IDS = tuple(
    sorted(
        task_id
        for task_id in list_all_envs()
        if task_id.startswith("MiniGrid-")
    )
)


def _debug_state(env: Any, env_id: int = 0) -> Any:
    debug_states = cast(Any, env)._debug_states(
        np.asarray([env_id], dtype=np.int32)
    )
    return debug_states[0]


def _render_array(env: Any, env_ids: Any = None) -> np.ndarray:
    frame = env.render(env_ids=env_ids)
    assert frame is not None
    return cast(np.ndarray, frame)


def _zero_action(num_envs: int) -> np.ndarray:
    return np.zeros((num_envs,), dtype=np.int32)


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
    if env_name in {
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


class MiniGridRenderTest(absltest.TestCase):
    """Render regression tests for MiniGrid environments."""

    def _oracle_frame(
        self, oracle: gym.Env[Any, Any], debug_state: Any, step_count: int
    ) -> np.ndarray:
        _patch_env_state(oracle.unwrapped, debug_state, step_count)
        return cast(
            np.ndarray,
            cast(Any, oracle.unwrapped).get_frame(
                highlight=False,
                tile_size=32,
                agent_pov=False,
            ),
        )

    def test_render_matches_upstream_oracle_for_multiple_steps_for_all_tasks(
        self,
    ) -> None:
        """Rendered frames should match the upstream MiniGrid oracle."""
        for task_id in _TASK_IDS:
            with self.subTest(task_id=task_id):
                env = make_gymnasium(
                    task_id, num_envs=1, render_mode="rgb_array"
                )
                oracle = gym.make(task_id, render_mode="rgb_array")
                try:
                    env.reset()
                    oracle.reset(seed=0)
                    step_count = 0
                    total = _RENDER_STEPS
                    for step_idx in range(total):
                        frame = _render_array(env)
                        frame_again = _render_array(env)
                        expected = self._oracle_frame(
                            oracle, _debug_state(env, 0), step_count
                        )

                        self.assertEqual(frame.shape, (1,) + expected.shape)
                        self.assertEqual(frame.dtype, np.uint8)
                        np.testing.assert_array_equal(frame[0], expected)
                        np.testing.assert_array_equal(frame, frame_again)
                        if step_idx + 1 < total:
                            env.step(_zero_action(1))
                            step_count += 1
                finally:
                    env.close()
                    oracle.close()

    def test_render_is_batch_consistent_and_state_invariant(self) -> None:
        """Rendering should be batch-consistent and free of side effects."""
        for task_id in _REPRESENTATIVE_TASK_IDS:
            with self.subTest(task_id=task_id):
                env = make_gymnasium(
                    task_id, num_envs=2, render_mode="rgb_array"
                )
                oracle0 = gym.make(task_id, render_mode="rgb_array")
                oracle1 = gym.make(task_id, render_mode="rgb_array")
                try:
                    env.reset()
                    oracle0.reset(seed=0)
                    oracle1.reset(seed=0)
                    step_count = 0
                    for step_idx in range(_RENDER_STEPS):
                        frame0 = _render_array(env)
                        frame1 = _render_array(env, env_ids=1)
                        frames = _render_array(env, env_ids=[0, 1])
                        frame0_again = _render_array(env)

                        expected0 = self._oracle_frame(
                            oracle0, _debug_state(env, 0), step_count
                        )
                        expected1 = self._oracle_frame(
                            oracle1, _debug_state(env, 1), step_count
                        )

                        self.assertEqual(frame0.shape, (1,) + expected0.shape)
                        self.assertEqual(frame1.shape, (1,) + expected1.shape)
                        self.assertEqual(frames.shape, (2,) + expected0.shape)
                        self.assertEqual(frame0.dtype, np.uint8)
                        self.assertEqual(frames.dtype, np.uint8)
                        np.testing.assert_array_equal(frame0[0], expected0)
                        np.testing.assert_array_equal(frame1[0], expected1)
                        np.testing.assert_array_equal(frame0[0], frames[0])
                        np.testing.assert_array_equal(frame1[0], frames[1])
                        np.testing.assert_array_equal(frame0, frame0_again)
                        if step_idx + 1 < _RENDER_STEPS:
                            env.step(_zero_action(2))
                            step_count += 1
                finally:
                    env.close()
                    oracle0.close()
                    oracle1.close()


if __name__ == "__main__":
    absltest.main()
