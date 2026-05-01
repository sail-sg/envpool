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
"""Tests for MarlGrid environments."""

from __future__ import annotations

import os
import re
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.marlgrid.registration  # noqa: F401
from envpool.registration import list_all_envs, make_gymnasium, make_spec

_TASK_CONFIGS = {
    "Goalcycle-demo-solo-v0": (1, 13, 7, 12, 3),
    "MarlGrid-1AgentCluttered15x15-v0": (1, 15, 5, 30, 3),
    "MarlGrid-2AgentEmpty9x9-v0": (2, 9, 7, 0, 3),
    "MarlGrid-3AgentCluttered11x11-v0": (3, 11, 7, 12, 3),
    "MarlGrid-3AgentCluttered15x15-v0": (3, 15, 7, 25, 3),
    "MarlGrid-3AgentEmpty9x9-v0": (3, 9, 7, 0, 3),
    "MarlGrid-4AgentEmpty9x9-v0": (4, 9, 7, 0, 3),
}

_COLOR_WALL = np.array([74, 65, 42], dtype=np.int16)
_COLOR_GOAL = np.array([0, 255, 0], dtype=np.int16)
_COLOR_BONUS = np.array([255, 255, 0], dtype=np.int16)


def _install_upstream_compat_modules() -> None:
    """Install small compatibility shims needed by the pinned MarlGrid source."""
    import gymnasium

    if not hasattr(np, "float"):
        np.__dict__["float"] = float
    if not hasattr(np, "bool"):
        np.__dict__["bool"] = np.bool_

    gym_mod = types.ModuleType("gym")
    gym_mod.Env = gymnasium.Env
    gym_mod.spaces = gymnasium.spaces
    gym_utils_mod = types.ModuleType("gym.utils")
    gym_seeding_mod = types.ModuleType("gym.utils.seeding")
    gym_seeding_mod.np_random = lambda seed=None: (
        np.random.RandomState(seed),
        seed,
    )
    gym_utils_mod.seeding = gym_seeding_mod
    gym_mod.utils = gym_utils_mod
    gym_envs_mod = types.ModuleType("gym.envs")
    gym_registration_mod = types.ModuleType("gym.envs.registration")
    gym_registration_mod.register = lambda *args, **kwargs: None
    gym_envs_mod.registration = gym_registration_mod
    gym_mod.envs = gym_envs_mod
    sys.modules.setdefault("gym", gym_mod)
    sys.modules.setdefault("gym.envs", gym_envs_mod)
    sys.modules.setdefault("gym.envs.registration", gym_registration_mod)
    sys.modules.setdefault("gym.utils", gym_utils_mod)
    sys.modules.setdefault("gym.utils.seeding", gym_seeding_mod)

    numba_mod = types.ModuleType("numba")
    numba_mod.boolean = np.bool_
    numba_mod.njit = _numba_njit
    sys.modules.setdefault("numba", numba_mod)

    rendering_mod = types.ModuleType("gym_minigrid.rendering")
    rendering_mod.fill_coords = _fill_coords
    rendering_mod.point_in_rect = _point_in_rect
    rendering_mod.point_in_triangle = _point_in_triangle
    rendering_mod.rotate_fn = _rotate_fn
    rendering_mod.downsample = _downsample
    rendering_mod.highlight_img = _highlight_img
    gym_minigrid_mod = types.ModuleType("gym_minigrid")
    gym_minigrid_mod.rendering = rendering_mod
    sys.modules.setdefault("gym_minigrid", gym_minigrid_mod)
    sys.modules.setdefault("gym_minigrid.rendering", rendering_mod)

    marlgrid_rendering_mod = types.ModuleType("marlgrid.rendering")

    class SimpleImageViewer:
        """Headless stand-in for upstream's human-mode viewer."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.isopen = False
            self.window = types.SimpleNamespace(set_caption=lambda *_: None)

        def imshow(self, *_: Any, **__: Any) -> None:
            return None

        def close(self) -> None:
            self.isopen = False

    marlgrid_rendering_mod.SimpleImageViewer = SimpleImageViewer
    sys.modules.setdefault("marlgrid.rendering", marlgrid_rendering_mod)


def _numba_njit(func: Any = None, **_: Any) -> Any:
    def decorator(inner: Any) -> Any:
        if inner.__name__ == "occlude_mask":
            return _occlude_mask
        return inner

    return decorator(func) if func is not None else decorator


def _occlude_mask(grid: np.ndarray, agent_pos: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(grid.shape[:2], dtype=np.bool_)
    ax, ay = agent_pos
    width, height = grid.shape[:2]
    mask[ax, ay] = True
    for y in range(min(ay + 1, height - 1), 0, -1):
        for x in range(ax, width):
            if mask[x, y] and grid[x, y]:
                if x < width - 1:
                    mask[x + 1, y] = True
                if y > 0:
                    mask[x, y - 1] = True
                    if x < width - 1:
                        mask[x + 1, y - 1] = True
        for x in range(min(ax + 1, width - 1), 0, -1):
            if mask[x, y] and grid[x, y]:
                if x > 0:
                    mask[x - 1, y] = True
                if y > 0:
                    mask[x, y - 1] = True
                    if x > 0:
                        mask[x - 1, y - 1] = True
    for y in range(ay, height):
        for x in range(ax, width):
            if mask[x, y] and grid[x, y]:
                if x < width - 1:
                    mask[x + 1, y] = True
                if y < height - 1:
                    mask[x, y + 1] = True
                    if x < width - 1:
                        mask[x + 1, y + 1] = True
        for x in range(min(ax + 1, width - 1), 0, -1):
            if mask[x, y] and grid[x, y]:
                if x > 0:
                    mask[x - 1, y] = True
                if y < height - 1:
                    mask[x, y + 1] = True
                    if x > 0:
                        mask[x - 1, y + 1] = True
    return mask


def _fill_coords(img: np.ndarray, fn: Any, color: np.ndarray) -> None:
    height, width = img.shape[:2]
    for y in range(height):
        yf = (y + 0.5) / height
        for x in range(width):
            xf = (x + 0.5) / width
            if fn(xf, yf):
                img[y, x] = color


def _point_in_rect(xmin: float, xmax: float, ymin: float, ymax: float) -> Any:
    return lambda x, y: xmin <= x <= xmax and ymin <= y <= ymax


def _point_in_triangle(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> Any:
    def fn(x: float, y: float) -> bool:
        v0 = np.array([c[0] - a[0], c[1] - a[1]])
        v1 = np.array([b[0] - a[0], b[1] - a[1]])
        v2 = np.array([x - a[0], y - a[1]])
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        inv = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv
        v = (dot00 * dot12 - dot01 * dot02) * inv
        return u >= 0 and v >= 0 and u + v < 1

    return fn


def _rotate_fn(fn: Any, cx: float, cy: float, theta: float) -> Any:
    def rotated(x: float, y: float) -> bool:
        x -= cx
        y -= cy
        xr = x * np.cos(-theta) - y * np.sin(-theta) + cx
        yr = y * np.cos(-theta) + x * np.sin(-theta) + cy
        return fn(xr, yr)

    return rotated


def _downsample(img: np.ndarray, factor: int) -> np.ndarray:
    height, width = img.shape[:2]
    return img.reshape(
        height // factor,
        factor,
        width // factor,
        factor,
        3,
    ).mean(axis=(1, 3))


def _highlight_img(
    img: np.ndarray,
    color: tuple[int, int, int] = (255, 255, 255),
    alpha: float = 0.30,
) -> None:
    blend = np.asarray(color, dtype=np.float32)
    img[:] = (img.astype(np.float32) * (1.0 - alpha) + blend * alpha).clip(
        0, 255
    )


def _upstream_registration_file() -> Path:
    runfiles = Path(os.environ["TEST_SRCDIR"])
    return runfiles / "marlgrid" / "marlgrid" / "envs" / "__init__.py"


def _upstream_registered_ids() -> list[str]:
    source = _upstream_registration_file().read_text(encoding="utf-8")
    return re.findall(r'register_marl_env\(\s*"([^"]+)"', source)


def _make_upstream_env(task_id: str) -> Any:
    _install_upstream_compat_modules()
    upstream_root = str(Path(os.environ["TEST_SRCDIR"]) / "marlgrid")
    if upstream_root not in sys.path:
        sys.path.insert(0, upstream_root)
    import marlgrid.envs as upstream_envs  # pylint: disable=import-outside-toplevel

    env_index = upstream_envs.registered_envs.index(task_id)
    return getattr(upstream_envs, f"env_{env_index}")()


def _player_order(info: dict[str, Any]) -> np.ndarray:
    players = info["players"]
    return np.lexsort((players["id"], players["env_id"]))


def _sort_players(value: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    return value[_player_order(info)]


def _sort_envs(value: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    return value[np.argsort(info["env_id"])]


def _action_for_player_order(
    canonical_action: np.ndarray,
    info: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    order = _player_order(info)
    action = np.empty_like(canonical_action)
    action[order] = canonical_action
    return {
        "players": {
            "env_id": info["players"]["env_id"],
            "action": action,
        },
    }


def _classify_tile(tile: np.ndarray) -> str | None:
    corners = np.array(
        [
            tile[1:5, 1:5].mean(axis=(0, 1)),
            tile[1:5, -5:-1].mean(axis=(0, 1)),
            tile[-5:-1, 1:5].mean(axis=(0, 1)),
            tile[-5:-1, -5:-1].mean(axis=(0, 1)),
        ],
        dtype=np.int16,
    )
    color = np.median(corners, axis=0)
    distances = {
        "wall": np.abs(color - _COLOR_WALL).sum(),
        "goal": np.abs(color - _COLOR_GOAL).sum(),
        "bonus": np.abs(color - _COLOR_BONUS).sum(),
    }
    name, distance = min(distances.items(), key=lambda item: item[1])
    return name if distance < 50 else None


def _sync_upstream_from_envpool(
    oracle_env: Any,
    frame: np.ndarray,
    info: dict[str, Any],
) -> None:
    from marlgrid.base import (  # pylint: disable=import-outside-toplevel
        MultiGrid,
    )
    from marlgrid.objects import (  # pylint: disable=import-outside-toplevel
        BonusTile,
        Goal,
        Wall,
    )

    grid_size = frame.shape[0] // 32
    oracle_env.grid = MultiGrid((grid_size, grid_size))
    bonus_id = 0
    for y in range(grid_size):
        for x in range(grid_size):
            tile = frame[y * 32 : (y + 1) * 32, x * 32 : (x + 1) * 32]
            tile_type = _classify_tile(tile)
            if tile_type == "wall":
                oracle_env.grid.set(x, y, Wall())
            elif tile_type == "goal":
                oracle_env.grid.set(x, y, Goal(color="green", reward=1))
            elif tile_type == "bonus":
                oracle_env.grid.set(
                    x,
                    y,
                    BonusTile(
                        color="yellow",
                        reward=1,
                        penalty=0.0,
                        bonus_id=bonus_id,
                        n_bonus=getattr(oracle_env, "n_bonus_tiles", 1),
                        initial_reward=getattr(
                            oracle_env, "initial_reward", True
                        ),
                        reset_on_mistake=getattr(
                            oracle_env,
                            "reset_on_mistake",
                            False,
                        ),
                    ),
                )
                bonus_id += 1

    order = _player_order(info)
    for index, agent_id in enumerate(info["players"]["id"][order]):
        agent = oracle_env.agents[int(agent_id)]
        agent.reset(new_episode=True)
        agent.dir = int(info["players"]["dir"][order][index])
        agent.done = bool(info["players"]["done"][order][index])
        agent.active = bool(info["players"]["active"][order][index])
        pos = info["players"]["pos"][order][index].astype(np.int64)
        agent.set_position((int(pos[0]), int(pos[1])))
        cell = oracle_env.grid.get(int(pos[0]), int(pos[1]))
        if cell is None:
            oracle_env.grid.set(int(pos[0]), int(pos[1]), agent)
        else:
            cell.agents.append(agent)
    oracle_env.step_count = 0


def _upstream_obs(oracle_env: Any) -> np.ndarray:
    return np.stack(oracle_env.gen_obs())


def _upstream_full_render(oracle_env: Any) -> np.ndarray:
    return oracle_env.grid.render(tile_size=32)


def _upstream_agent_info(oracle_env: Any) -> dict[str, np.ndarray]:
    return {
        "done": np.array(
            [agent.done for agent in oracle_env.agents], dtype=np.bool_
        ),
        "active": np.array(
            [agent.active for agent in oracle_env.agents],
            dtype=np.bool_,
        ),
        "pos": np.array(
            [agent.pos for agent in oracle_env.agents], dtype=np.int32
        ),
        "dir": np.array(
            [agent.dir for agent in oracle_env.agents], dtype=np.int32
        ),
    }


class MarlGridTest(absltest.TestCase):
    """Tests for MarlGrid registration, determinism, and rendering."""

    def test_registry_coverage(self) -> None:
        """Validate registered EnvPool task configs."""
        task_ids = sorted(
            task_id
            for task_id in list_all_envs()
            if task_id.startswith(("MarlGrid-", "Goalcycle-"))
        )
        self.assertEqual(task_ids, sorted(_TASK_CONFIGS))
        for task_id, (
            n_agents,
            grid_size,
            view_size,
            n_clutter,
            n_bonus_tiles,
        ) in _TASK_CONFIGS.items():
            spec = make_spec(task_id)
            self.assertEqual(spec.config.n_agents, n_agents)
            self.assertEqual(spec.config.max_num_players, n_agents)
            self.assertEqual(spec.config.grid_size, grid_size)
            self.assertEqual(spec.config.view_size, view_size)
            self.assertEqual(spec.config.n_clutter, n_clutter)
            self.assertEqual(spec.config.n_bonus_tiles, n_bonus_tiles)

    def test_registry_matches_pinned_upstream(self) -> None:
        """Check EnvPool tasks against the pinned upstream registry source."""
        self.assertEqual(
            sorted(_upstream_registered_ids()), sorted(_TASK_CONFIGS)
        )

    def test_multiplayer_step_shapes(self) -> None:
        """Validate player-shaped reset and step outputs."""
        env = make_gymnasium(
            "MarlGrid-3AgentCluttered11x11-v0",
            num_envs=2,
            batch_size=2,
            seed=0,
        )
        try:
            obs, info = env.reset()
            self.assertEqual(obs.shape, (6, 56, 56, 3))
            np.testing.assert_array_equal(
                _sort_players(info["players"]["env_id"], info),
                np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
            )
            np.testing.assert_array_equal(
                _sort_players(info["players"]["id"], info),
                np.array([0, 1, 2, 0, 1, 2], dtype=np.int32),
            )
            action = {
                "players": {
                    "env_id": info["players"]["env_id"],
                    "action": np.full((6,), 2, dtype=np.int32),
                },
            }
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertEqual(obs.shape, (6, 56, 56, 3))
            self.assertEqual(reward.shape, (6,))
            self.assertEqual(terminated.shape, (2,))
            self.assertEqual(truncated.shape, (2,))
            self.assertEqual(info["players"]["done"].shape, (6,))
        finally:
            env.close()

    def test_same_seed_is_deterministic(self) -> None:
        """Check same-seed rollouts independent of async return order."""
        env0 = make_gymnasium("MarlGrid-4AgentEmpty9x9-v0", num_envs=2, seed=7)
        env1 = make_gymnasium("MarlGrid-4AgentEmpty9x9-v0", num_envs=2, seed=7)
        actions = [
            np.array([0, 1, 2, 2, 0, 1, 2, 2], dtype=np.int32),
            np.array([2, 2, 1, 0, 2, 2, 1, 0], dtype=np.int32),
            np.array([1, 2, 2, 0, 1, 2, 2, 0], dtype=np.int32),
        ]
        try:
            obs0, info0 = env0.reset()
            obs1, info1 = env1.reset()
            np.testing.assert_array_equal(
                _sort_players(obs0, info0), _sort_players(obs1, info1)
            )
            np.testing.assert_array_equal(
                _sort_players(info0["players"]["pos"], info0),
                _sort_players(info1["players"]["pos"], info1),
            )
            for action in actions:
                out0 = env0.step(_action_for_player_order(action, info0))
                out1 = env1.step(_action_for_player_order(action, info1))
                obs0, reward0, terminated0, truncated0, info0 = out0
                obs1, reward1, terminated1, truncated1, info1 = out1
                np.testing.assert_array_equal(
                    _sort_players(obs0, info0), _sort_players(obs1, info1)
                )
                np.testing.assert_array_equal(
                    _sort_players(reward0, info0), _sort_players(reward1, info1)
                )
                np.testing.assert_array_equal(
                    _sort_envs(terminated0, info0),
                    _sort_envs(terminated1, info1),
                )
                np.testing.assert_array_equal(
                    _sort_envs(truncated0, info0), _sort_envs(truncated1, info1)
                )
                np.testing.assert_array_equal(
                    _sort_players(info0["players"]["pos"], info0),
                    _sort_players(info1["players"]["pos"], info1),
                )
        finally:
            env0.close()
            env1.close()

    def test_aligns_with_pinned_upstream_after_reset_sync(self) -> None:
        """Compare step outputs with the pinned upstream source after reset sync."""
        for task_id, (n_agents, grid_size, _, _, _) in _TASK_CONFIGS.items():
            with self.subTest(task_id=task_id):
                env = make_gymnasium(
                    task_id,
                    num_envs=1,
                    seed=11,
                    render_mode="rgb_array",
                )
                oracle_env = _make_upstream_env(task_id)
                try:
                    obs, info = env.reset()
                    frame = env.render()
                    self.assertIsNotNone(frame)
                    assert frame is not None
                    frame0 = frame[0]
                    _sync_upstream_from_envpool(oracle_env, frame0, info)

                    np.testing.assert_array_equal(
                        _sort_players(obs, info),
                        _upstream_obs(oracle_env),
                    )
                    np.testing.assert_array_equal(
                        frame0,
                        _upstream_full_render(oracle_env),
                    )

                    actions = [
                        (np.arange(n_agents, dtype=np.int32) + 0) % 3,
                        (np.arange(n_agents, dtype=np.int32) + 1) % 3,
                        (np.arange(n_agents, dtype=np.int32) + 2) % 3,
                        np.full(n_agents, 0, dtype=np.int32),
                        np.full(n_agents, 1, dtype=np.int32),
                    ]
                    if task_id.startswith("Goalcycle-"):
                        actions = [
                            np.full(n_agents, 0, dtype=np.int32),
                            np.full(n_agents, 1, dtype=np.int32),
                            np.full(n_agents, 6, dtype=np.int32),
                        ]
                    for action in actions:
                        obs, reward, terminated, truncated, info = env.step({
                            "players": {
                                "env_id": np.zeros(n_agents, dtype=np.int32),
                                "action": action,
                            },
                        })
                        oracle_obs, oracle_reward, oracle_done, _ = (
                            oracle_env.step(action.tolist())
                        )
                        np.testing.assert_array_equal(
                            _sort_players(obs, info),
                            np.stack(oracle_obs),
                        )
                        np.testing.assert_array_equal(
                            _sort_players(reward, info),
                            oracle_reward,
                        )
                        self.assertEqual(
                            bool(terminated[0] or truncated[0]),
                            bool(oracle_done),
                        )
                        oracle_info = _upstream_agent_info(oracle_env)
                        for key, value in oracle_info.items():
                            np.testing.assert_array_equal(
                                _sort_players(info["players"][key], info),
                                value,
                            )
                        frame = env.render()
                        self.assertIsNotNone(frame)
                        assert frame is not None
                        np.testing.assert_array_equal(
                            frame[0],
                            _upstream_full_render(oracle_env),
                        )
                    self.assertEqual(
                        frame0.shape, (grid_size * 32, grid_size * 32, 3)
                    )
                finally:
                    env.close()

    def test_render_rgb_array_all_tasks(self) -> None:
        """Render every registered task after reset and after multiple steps."""
        for task_id, (_, grid_size, _, _, _) in _TASK_CONFIGS.items():
            with self.subTest(task_id=task_id):
                env = make_gymnasium(
                    task_id,
                    num_envs=2,
                    seed=3,
                    render_mode="rgb_array",
                )
                try:
                    _, info = env.reset()
                    native_size = grid_size * 32
                    frame = env.render(np.array([0, 1], dtype=np.int32))
                    self.assertIsNotNone(frame)
                    assert frame is not None
                    self.assertEqual(frame.dtype, np.uint8)
                    self.assertEqual(
                        frame.shape, (2, native_size, native_size, 3)
                    )
                    self.assertGreater(int(frame.sum()), 0)
                    for action_id in (2, 1):
                        action = {
                            "players": {
                                "env_id": info["players"]["env_id"],
                                "action": np.full(
                                    info["players"]["env_id"].shape,
                                    action_id,
                                    dtype=np.int32,
                                ),
                            },
                        }
                        _, _, _, _, info = env.step(action)
                    frame = env.render(np.array([1, 0], dtype=np.int32))
                    self.assertEqual(frame.dtype, np.uint8)
                    self.assertEqual(
                        frame.shape, (2, native_size, native_size, 3)
                    )
                    self.assertGreater(int(frame.sum()), 0)
                finally:
                    env.close()


if __name__ == "__main__":
    absltest.main()
