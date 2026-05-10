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
"""Registry and smoke tests for the native Jumanji family."""

# ruff: noqa: D102

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration as jumanji_registration
from envpool.registration import list_all_envs, make_gymnasium, make_spec

_OFFICIAL_JUMANJI_V111_IDS = (
    "BinPack-v2",
    "CVRP-v1",
    "Cleaner-v0",
    "Connector-v2",
    "FlatPack-v0",
    "Game2048-v1",
    "GraphColoring-v1",
    "JobShop-v0",
    "Knapsack-v1",
    "LevelBasedForaging-v0",
    "MMST-v0",
    "Maze-v0",
    "Minesweeper-v0",
    "MultiCVRP-v0",
    "PacMan-v1",
    "RobotWarehouse-v0",
    "RubiksCube-partly-scrambled-v0",
    "RubiksCube-v0",
    "SearchAndRescue-v0",
    "SlidingTilePuzzle-v0",
    "Snake-v1",
    "Sokoban-v0",
    "Sudoku-v0",
    "Sudoku-very-easy-v0",
    "TSP-v1",
    "Tetris-v0",
)


def _zeros_action(space: gymnasium.Space[Any], num_envs: int) -> np.ndarray:
    if isinstance(space, gymnasium.spaces.Discrete):
        return np.zeros((num_envs,), dtype=np.int32)
    if isinstance(space, gymnasium.spaces.Box):
        return np.zeros((num_envs, *space.shape), dtype=space.dtype)
    raise TypeError(f"unsupported Jumanji action space: {space!r}")


def _assert_obs_batch(obs: Any, num_envs: int) -> None:
    if isinstance(obs, dict):
        for value in obs.values():
            _assert_obs_batch(value, num_envs)
        return
    arr = np.asarray(obs)
    assert arr.shape[0] == num_envs, arr.shape


def _assert_render_batch(frame: np.ndarray | None, num_envs: int) -> None:
    assert frame is not None
    assert frame.shape == (num_envs, 256, 256, 3), frame.shape
    assert frame.dtype == np.uint8, frame.dtype
    for env_id in range(num_envs):
        env_frame = frame[env_id]
        assert int(np.max(env_frame)) > int(np.min(env_frame)), env_id


class JumanjiRegistryTest(absltest.TestCase):
    """Smoke and registry coverage for the native Jumanji bindings."""

    def test_registered_ids_match_pinned_oracle_list(self) -> None:
        """Checks EnvPool exposes every pinned Jumanji v1.1.1 ID."""
        self.assertEqual(
            tuple(jumanji_registration.jumanji_env_ids),
            _OFFICIAL_JUMANJI_V111_IDS,
        )
        registered = set(list_all_envs())
        for task_id in _OFFICIAL_JUMANJI_V111_IDS:
            self.assertIn(task_id, registered)
            self.assertIn(f"Jumanji/{task_id}", registered)

    def test_make_reset_step_and_render_all_registered_ids(self) -> None:
        """Checks each Jumanji ID can be constructed and stepped."""
        for task_id in _OFFICIAL_JUMANJI_V111_IDS:
            with self.subTest(task_id=task_id):
                spec = make_spec(task_id, num_envs=2, seed=0)
                self.assertGreater(int(spec.config.max_episode_steps), 0)
                env = make_gymnasium(
                    task_id, num_envs=2, seed=0, render_mode="rgb_array"
                )
                try:
                    obs, info = env.reset()
                    _assert_obs_batch(obs, 2)
                    self.assertEqual(info["env_id"].tolist(), [0, 1])
                    reset_frame = env.render(
                        env_ids=np.asarray([0, 1], dtype=np.int32)
                    )
                    _assert_render_batch(reset_frame, 2)
                    action = _zeros_action(env.action_space, 2)
                    obs, reward, terminated, truncated, info = env.step(action)
                    _assert_obs_batch(obs, 2)
                    self.assertEqual(np.asarray(reward).shape, (2,))
                    self.assertEqual(np.asarray(terminated).shape, (2,))
                    self.assertEqual(np.asarray(truncated).shape, (2,))
                    self.assertEqual(info["env_id"].tolist(), [0, 1])
                    frame = env.render(
                        env_ids=np.asarray([0, 1], dtype=np.int32)
                    )
                    _assert_render_batch(frame, 2)
                finally:
                    env.close()

    def test_render_cache_handles_positional_recv_reset(self) -> None:
        env = make_gymnasium(
            "Game2048-v1", num_envs=1, seed=0, render_mode="rgb_array"
        )
        try:
            env.async_reset()
            _, info = env.recv(True)
            self.assertEqual(info["env_id"].tolist(), [0])
            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))
            _assert_render_batch(frame, 1)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
