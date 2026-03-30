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
from minigrid.core.world_object import WorldObj

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
_REPRESENTATIVE_TASK_IDS = (
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-BlockedUnlockPickup-v0",
)
_TASK_IDS = tuple(
    sorted(task_id for task_id in list_all_envs() if task_id.startswith("MiniGrid-"))
)


def _debug_state(env: Any, env_id: int = 0) -> Any:
    debug_states = cast(
        Any, env
    )._debug_states(np.asarray([env_id], dtype=np.int32))
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
    carrying.cur_pos = None
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

    def _oracle_frame(self, task_id: str, debug_state: Any, step_count: int) -> np.ndarray:
        oracle = gym.make(task_id, render_mode="rgb_array")
        try:
            oracle.reset(seed=0)
            _patch_env_state(oracle.unwrapped, debug_state, step_count)
            return cast(
                np.ndarray,
                oracle.unwrapped.get_frame(
                    highlight=False,
                    tile_size=32,
                    agent_pov=False,
                ),
            )
        finally:
            oracle.close()

    def test_render_matches_upstream_oracle_first_frame_for_all_tasks(self) -> None:
        """The first rendered frame should match the upstream MiniGrid oracle."""
        for task_id in _TASK_IDS:
            with self.subTest(task_id=task_id):
                env = make_gymnasium(task_id, num_envs=1, render_mode="rgb_array")
                try:
                    env.reset()
                    frame = env.render()
                    expected = self._oracle_frame(task_id, _debug_state(env, 0), 0)

                    self.assertEqual(frame.shape, (1,) + expected.shape)
                    self.assertEqual(frame.dtype, np.uint8)
                    np.testing.assert_array_equal(frame[0], expected)
                finally:
                    env.close()

    def test_render_is_batch_consistent_and_state_invariant(self) -> None:
        """Rendering should be batch-consistent and free of side effects."""
        for task_id in _REPRESENTATIVE_TASK_IDS:
            with self.subTest(task_id=task_id):
                env = make_gymnasium(task_id, num_envs=2, render_mode="rgb_array")
                try:
                    env.reset()
                    frame0 = env.render()
                    frame1 = env.render(env_ids=1)
                    frames = env.render(env_ids=[0, 1])
                    frame0_again = env.render()

                    expected0 = self._oracle_frame(task_id, _debug_state(env, 0), 0)

                    self.assertEqual(frame0.shape, (1,) + expected0.shape)
                    self.assertEqual(frame1.shape, (1,) + expected0.shape)
                    self.assertEqual(frames.shape, (2,) + expected0.shape)
                    self.assertEqual(frame0.dtype, np.uint8)
                    self.assertEqual(frames.dtype, np.uint8)
                    np.testing.assert_array_equal(frame0[0], expected0)
                    np.testing.assert_array_equal(frame0[0], frames[0])
                    np.testing.assert_array_equal(frame1[0], frames[1])
                    np.testing.assert_array_equal(frame0, frame0_again)

                    env.step(np.asarray([0, 0], dtype=np.int32))
                    stepped0 = env.render()
                    stepped_frames = env.render(env_ids=[0, 1])
                    stepped0_again = env.render()
                    expected_after = self._oracle_frame(
                        task_id, _debug_state(env, 0), 1
                    )

                    np.testing.assert_array_equal(stepped0[0], expected_after)
                    np.testing.assert_array_equal(stepped0[0], stepped_frames[0])
                    np.testing.assert_array_equal(stepped0, stepped0_again)
                finally:
                    env.close()


if __name__ == "__main__":
    absltest.main()
