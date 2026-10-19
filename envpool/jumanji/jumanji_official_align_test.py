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
"""Independent alignment tests against the pinned official Jumanji oracle."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any
from unittest.mock import patch

# Configure the test-only backend before importing JAX on Windows.
from envpool.jumanji.oracle import (
    _as_numpy,
    _envpool_kwargs_from_official_state,
    _find_viewer,
    _make_official_env,
    _minesweeper_flat_mines,
    _official_render,
)

# isort: split

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from absl.testing import absltest, parameterized  # noqa: E402
from jumanji.environments.logic.game_2048.utils import move  # noqa: E402
from jumanji.environments.packing.tetris import (
    utils as tetris_utils,  # noqa: E402
)
from jumanji.environments.routing.pac_man import utils as pacman_utils
from jumanji.environments.routing.snake.types import Position  # noqa: E402
from matplotlib import colormaps  # noqa: E402

import envpool.jumanji.registration as jumanji_registration  # noqa: E402
from envpool.registration import make_gymnasium  # noqa: E402

_RENDER_ALIGN_STEPS = 3
_RENDER_SIZE = 128


def _tree_to_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _tree_to_dict(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {
            key: _tree_to_dict(getattr(value, key)) for key in value._fields
        }
    if is_dataclass(value):
        return {
            field.name: _tree_to_dict(getattr(value, field.name))
            for field in fields(value)
        }
    return _as_numpy(value)


def _first_env_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _first_env_tree(item) for key, item in value.items()}
    array = np.asarray(value)
    assert array.shape[:1] == (1,), array.shape
    return array[0]


def _project(expected: Any, actual: Any) -> Any:
    if isinstance(actual, dict):
        assert isinstance(expected, dict), (expected, actual)
        return {key: _project(expected[key], actual[key]) for key in actual}
    return expected


def _assert_tree_values(actual: Any, expected: Any, label: str) -> None:
    if isinstance(actual, dict):
        assert isinstance(expected, dict), label
        assert actual.keys() == expected.keys(), label
        for key in actual:
            if key == "ems" and "ems_mask" in actual:
                # Inactive EMS slots are scratch storage in upstream; only the
                # mask and active coordinates/order participate in the task.
                mask = np.asarray(expected["ems_mask"], dtype=bool)
                for axis in actual[key]:
                    assert actual[key][axis].shape == expected[key][axis].shape
                    _assert_tree_values(
                        actual[key][axis][mask],
                        expected[key][axis][mask],
                        f"{label}.ems.{axis}",
                    )
            else:
                _assert_tree_values(
                    actual[key], expected[key], f"{label}.{key}"
                )
        return
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.shape == expected_array.shape, label
    if np.issubdtype(expected_array.dtype, np.floating):
        if label.startswith("Snake-v1 ") and label.endswith("obs.grid"):
            # Body/head/tail/fruit channels encode discrete occupancy.
            np.testing.assert_array_equal(
                actual_array[..., :4], expected_array[..., :4], err_msg=label
            )
        atol = 1e-6
        if label.startswith("SearchAndRescue-v0 "):
            # Float32 integration of the same checked velocity. Across
            # three 400-step macOS/Linux rollouts: positions <= 3.61e-6 and
            # normalized view distances <= 2.93e-5; visibility stayed exact.
            if label.endswith("positions"):
                atol = 5e-6
            elif label.endswith("searcher_views"):
                np.testing.assert_array_equal(
                    actual_array == -1, expected_array == -1, err_msg=label
                )
                atol = 5e-5
        np.testing.assert_allclose(
            actual_array, expected_array, rtol=1e-6, atol=atol, err_msg=label
        )
    else:
        np.testing.assert_array_equal(
            actual_array, expected_array, err_msg=label
        )
    # Jumanji 1.1.2 declares SearchAndRescue.step as int32, emits int64 on
    # reset, and int32 after step. EnvPool keeps the public spec dtype stable.
    if label.startswith("SearchAndRescue-v0 ") and label.endswith("obs.step"):
        return
    if label.startswith("PacMan-v1 ") and (
        label.endswith(("obs.player_locations.x", "obs.player_locations.y"))
    ):
        return
    assert actual_array.dtype == expected_array.dtype, (
        f"{label} dtype mismatch: {actual_array.dtype} != {expected_array.dtype}"
    )


def _tetromino_index(env: Any, obs: dict) -> int:
    matches = np.all(
        np.asarray(env.TETROMINOES_LIST[:, 0]) == obs["tetromino"][0],
        axis=(1, 2),
    )
    assert matches.sum() == 1
    return int(matches.argmax())


def _fruit_position(obs: dict) -> Any:
    locations = np.argwhere(obs["grid"][0, ..., 3] == 1)
    assert locations.shape == (1, 2)
    return jnp.asarray(locations[0], dtype=jnp.int32)


def _sampled_reset(
    task: str, env: Any, state: Any, timestep: Any, obs: dict
) -> tuple[Any, Any]:
    if task == "Game2048-v1":
        board = jnp.asarray(obs["board"][0])
        nonzero = np.asarray(board)[np.asarray(board) != 0]
        assert nonzero.shape == (1,) and nonzero[0] in (1, 2)
        mask = env._get_action_mask(board)
        state = state.replace(board=board, action_mask=mask)
        observation = timestep.observation._replace(
            board=board, action_mask=mask
        )
    elif task == "Snake-v1":
        grid = obs["grid"][0]
        heads = np.argwhere(grid[..., 1] == 1)
        assert heads.shape == (1, 2)
        body = jnp.asarray(grid[..., 0], dtype=bool)
        assert body.sum() == 1
        np.testing.assert_array_equal(grid[..., 0], grid[..., 1])
        np.testing.assert_array_equal(grid[..., 0], grid[..., 2])
        np.testing.assert_array_equal(grid[..., 0], grid[..., 4])
        head = Position(*jnp.asarray(heads[0], dtype=jnp.int32))
        fruit = Position(*_fruit_position(obs))
        assert not body[tuple(fruit)]
        body_state = body.astype(jnp.int32)
        state = state.replace(
            body=body,
            body_state=body_state,
            head_position=head,
            tail=body,
            fruit_position=fruit,
            length=jnp.array(1),
            action_mask=env._get_action_mask(head, body_state),
        )
        observation = env._state_to_observation(state)
    else:
        index = jnp.asarray(_tetromino_index(env, obs), dtype=jnp.int32)
        piece = env.TETROMINOES_LIST[index, 0]
        mask = env._calculate_action_mask(state.grid_padded, index)
        state = state.replace(
            tetromino_index=index, new_tetromino=piece, action_mask=mask
        )
        observation = timestep.observation._replace(
            tetromino=piece, action_mask=mask
        )
    return state, timestep.replace(observation=observation)


def _sampled_step(task: str, env: Any) -> Any:
    if task == "Game2048-v1":

        @jax.jit
        def place_tile(state: Any, action: Any, cell: Any, value: Any) -> Any:
            def add(board: Any, _key: Any) -> Any:
                return (
                    board.reshape(-1).at[cell].set(value).reshape(board.shape)
                )

            with patch.object(env, "_add_random_cell", add):
                return env.step(state, action)

        moved = jax.jit(move)

        def step(state: Any, action: Any, obs: dict, _info: dict) -> Any:
            board, _ = moved(state.board, action)
            actual = obs["board"][0]
            changed = np.flatnonzero(
                np.asarray(board).ravel() != actual.ravel()
            )
            if bool(state.action_mask[action]):
                assert changed.size == 1
                cell = int(changed[0])
                value = int(actual.ravel()[cell])
                assert np.asarray(board).ravel()[cell] == 0 and value in (1, 2)
            else:
                assert changed.size == 0
                cell, value = 0, 1
            return place_tile(state, action, cell, value)

        return step
    elif task == "Snake-v1":

        @jax.jit
        def move_snake(state: Any, action: Any, fruit: Any) -> Any:
            with patch.object(
                env, "_sample_fruit_coord", lambda *_: Position(*fruit)
            ):
                return env.step(state, action)

        def snake_step(state: Any, action: Any, obs: dict, _info: dict) -> Any:
            fruit = _fruit_position(obs)
            result, timestep = move_snake(state, action, fruit)
            if not bool(jnp.all(result.body)):
                assert not result.body[tuple(fruit)]
            return result, timestep

        return snake_step
    else:

        @jax.jit
        def drop_piece(state: Any, action: Any, index: Any) -> Any:
            with patch.object(
                tetris_utils,
                "sample_tetromino_list",
                lambda _key, pieces: (pieces[index, 0], index),
            ):
                return env.step(state, action)

        def tetris_step(state: Any, action: Any, obs: dict, _info: dict) -> Any:
            return drop_piece(
                state,
                action,
                jnp.asarray(_tetromino_index(env, obs), dtype=jnp.int32),
            )

    return tetris_step


def _pacman_step(env: Any) -> Any:
    original = pacman_utils.check_ghost_wall_collisions

    @jax.jit
    def allowed(state: Any, action: Any) -> Any:
        state = state.replace(last_direction=action)
        options = []
        for ghost in range(4):
            _, walls, candidates = original(
                state,
                state.ghost_locations[ghost],
                ghost,
                state.player_locations,
                state.ghost_init_targets[ghost],
                state.old_ghost_locations[ghost],
                state.scatter_targets[ghost],
                env.x_size,
                env.y_size,
            )
            tunnel = jnp.all(walls == jnp.array([1, 0, 1, 0])) | jnp.all(
                walls == jnp.array([0, 1, 0, 1])
            )
            choices = jnp.where(
                tunnel,
                jax.nn.one_hot(state.ghost_actions[ghost], 5, dtype=bool),
                jnp.r_[candidates.astype(bool), False],
            )
            options.append(
                jnp.where(
                    state.ghost_starts[ghost] < 0,
                    choices,
                    jax.nn.one_hot(4, 5, dtype=bool),
                )
            )
        return jnp.stack(options)

    @jax.jit
    def transition(state: Any, action: Any, choices: Any) -> Any:
        def selected(*args: Any) -> Any:
            collision, walls, _ = original(*args)
            # Restrict the independently validated candidate set to the actual
            # native ghost action. Upstream still performs movement/collisions.
            return (
                collision,
                walls,
                jax.nn.one_hot(choices[args[2]], 4, dtype=jnp.int32),
            )

        with patch.object(
            pacman_utils, "check_ghost_wall_collisions", selected
        ):
            return env.step(state, action)

    def step(state: Any, action: Any, obs: dict, info: dict) -> Any:
        choices = np.asarray(info["ghost_actions"][0])
        assert choices.shape == (4,) and np.all((0 <= choices) & (choices <= 4))
        assert np.asarray(allowed(state, action))[np.arange(4), choices].all()
        return transition(state, action, jnp.asarray(choices))

    return step


def _mmst_step(env: Any) -> Any:
    # Three agents have only six possible priority orders. Check the native
    # outcome against the complete official transition set, not one RNG stream.
    permutations: dict[tuple[int, ...], Any] = {}
    for seed in range(100):
        key = jax.random.PRNGKey(seed)
        order = tuple(
            np.asarray(
                jax.random.permutation(jax.random.split(key)[1], 3)
            ).tolist()
        )
        permutations.setdefault(order, key)
        if len(permutations) == 6:
            break
    assert len(permutations) == 6
    keys = jnp.stack(list(permutations.values()))

    @jax.jit
    def transitions(state: Any, action: Any) -> Any:
        return jax.vmap(lambda key: env.step(state.replace(key=key), action))(
            keys
        )

    def step(state: Any, action: Any, obs: dict, info: dict) -> Any:
        states, timesteps = transitions(state, action)
        matches = np.flatnonzero(
            np.all(np.asarray(states.positions) == obs["positions"][0], axis=1)
        )
        assert matches.size, (
            "native MMST move does not match any legal priority order"
        )
        index = int(matches[0])
        return jax.tree_util.tree_map(
            lambda value: value[index], (states, timesteps)
        )

    return step


def _native_reset_reference(
    task_id: str,
    official: Any,
    state: Any,
    timestep: Any,
    obs: dict,
    info: dict,
) -> tuple[Any, Any]:
    """Use actual native reset state when RNG streams need not agree."""
    if task_id in ("Game2048-v1", "Snake-v1", "Tetris-v0"):
        return _sampled_reset(task_id, official, state, timestep, obs)
    if task_id != "SearchAndRescue-v0":
        return state, timestep
    for value in (obs["positions"][0], info["target_positions"][0]):
        assert np.all(np.isfinite(value)) and np.all((value >= 0) & (value < 1))
    headings = info["searcher_headings"][0]
    speeds = info["searcher_speeds"][0]
    assert np.all((headings >= 0) & (headings < 2 * np.pi))
    assert np.all((speeds >= 0.005) & (speeds <= 0.02))
    np.testing.assert_array_equal(info["target_velocities"], 0)
    np.testing.assert_array_equal(info["target_found"], False)
    state = state.replace(
        searchers=state.searchers.replace(
            pos=jax.numpy.asarray(obs["positions"][0]),
            heading=jax.numpy.asarray(headings),
            speed=jax.numpy.asarray(speeds),
        ),
        targets=state.targets.replace(
            pos=jax.numpy.asarray(info["target_positions"][0]),
            vel=jax.numpy.asarray(info["target_velocities"][0]),
            found=jax.numpy.asarray(info["target_found"][0]),
        ),
    )
    return state, timestep.replace(
        observation=official._state_to_observation(state)
    )


def _native_step_reference(task_id: str, official: Any) -> Any:
    """Condition target motion on checked native velocity, not shared draws."""
    if task_id == "PacMan-v1":
        return _pacman_step(official)
    if task_id == "MMST-v0":
        return _mmst_step(official)
    if task_id in ("Game2048-v1", "Snake-v1", "Tetris-v0"):
        return _sampled_step(task_id, official)
    if task_id == "SearchAndRescue-v0":
        # Gaussian increments and clipping have independent distribution tests.
        # The official dynamics still integrates positions and the task still
        # computes visibility, detection, reward and termination independently.
        official._target_dynamics.acc_std = 0.0
    compiled = jax.jit(official.step)

    def step(
        state: Any, action: Any, _obs: dict, info: dict
    ) -> tuple[Any, Any]:
        if task_id != "SearchAndRescue-v0":
            return compiled(state, action)
        velocity = info["target_velocities"][0]
        assert np.all(np.isfinite(velocity))
        assert np.all(
            np.linalg.norm(velocity, axis=-1)
            <= official._target_dynamics.vel_max + 1e-9
        )
        state = state.replace(
            targets=state.targets.replace(vel=jax.numpy.asarray(velocity))
        )
        result, timestep = compiled(state, action)
        np.testing.assert_allclose(
            info["target_positions"][0],
            result.targets.pos,
            rtol=1e-6,
            atol=5e-6,
        )
        np.testing.assert_array_equal(
            info["target_found"][0], result.targets.found
        )
        return result, timestep

    return step


def _envpool_action(action: Any) -> np.ndarray:
    action_array = _as_numpy(action)
    return action_array.reshape((1, *action_array.shape))


def _assert_reward_values(
    actual: Any,
    expected: Any,
    info: dict[str, Any],
    task_id: str,
    label: str,
) -> None:
    expected = _as_numpy(expected)
    if task_id == "Connector-v3":
        np.testing.assert_allclose(
            info["agent_rewards"][0],
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg=label,
        )
        expected = np.float32(expected.mean(dtype=np.float64))
    if task_id == "SearchAndRescue-v0":
        np.testing.assert_allclose(
            info["searcher_rewards"][0],
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg=label,
        )
        expected = expected.sum(dtype=np.float32)
    np.testing.assert_allclose(
        np.asarray(actual)[0], expected, rtol=1e-6, atol=1e-6, err_msg=label
    )


def _first_valid_action(
    official_env: Any,
    observation: Any,
    task_id: str | None = None,
    official_state: Any | None = None,
    rng: np.random.Generator | None = None,
) -> Any:
    spec = official_env.action_spec
    action = _as_numpy(spec.generate_value())
    choose = lambda values: values[0] if rng is None else rng.choice(values)
    if rng is not None:
        if np.issubdtype(action.dtype, np.floating):
            action = rng.uniform(
                spec.minimum, spec.maximum, size=action.shape
            ).astype(action.dtype)
        else:
            action = rng.integers(
                spec.minimum,
                spec.maximum + 1,
                size=action.shape,
                dtype=action.dtype,
            )
    if task_id == "Minesweeper-v0" and official_state is not None:
        board = _as_numpy(observation.board).reshape(-1)
        mines = _minesweeper_flat_mines(official_state)
        safe = [
            location
            for location, value in enumerate(board)
            if int(value) == -1 and location not in mines
        ]
        if safe:
            location = choose(safe)
            return np.asarray(
                [location // 10, location % 10], dtype=action.dtype
            )
    if np.issubdtype(action.dtype, np.floating):
        return np.zeros_like(action) if rng is None else action
    if not hasattr(observation, "action_mask"):
        return action

    mask = _as_numpy(observation.action_mask).astype(bool, copy=False)
    if not np.any(mask):
        return action

    if action.shape == () and mask.ndim == 1:
        return np.asarray(choose(np.flatnonzero(mask)), dtype=action.dtype)

    if action.shape == mask.shape[:-1]:
        selected = np.zeros_like(action)
        for prefix in np.ndindex(mask.shape[:-1]):
            valid = np.flatnonzero(mask[prefix])
            selected[prefix] = choose(valid) if valid.size else action[prefix]
        return selected

    if action.ndim == 1 and action.shape[0] == mask.ndim:
        return np.asarray(
            np.argwhere(mask)[choose(np.arange(np.count_nonzero(mask)))],
            dtype=action.dtype,
        )

    return action


def _assert_render_content(
    image: np.ndarray, state: dict[str, Any], task_id: str
) -> None:
    """Check semantic landmarks from oracle state across native drawing styles."""
    # Small pixel neighborhoods cover coordinate rounding. A 12/255 color budget
    # stays below the palette separations; neither rule compares exact pixels.
    colors: Any
    xy: Any
    height, width, _ = image.shape
    palette = (
        colormaps["tab20"](np.arange(20))[:, :3][
            list(range(0, 20, 2)) + list(range(1, 20, 2))
        ]
        * 255
    )
    white = (255, 255, 255)

    def color_at(point: Any, color: Any, radius: int = 1) -> None:
        x, y = np.rint(point).astype(int)
        patch = image[
            max(0, y - radius) : min(height, y + radius + 1),
            max(0, x - radius) : min(width, x + radius + 1),
        ].astype(float)
        assert patch.size and np.any(
            np.max(np.abs(patch - color), axis=-1) <= 12
        ), (task_id, "missing landmark", point, color, patch.tolist())

    def center(row: Any, col: Any, shape: Any) -> tuple[float, float]:
        return ((col + 0.5) * width / shape[1], (row + 0.5) * height / shape[0])

    def cell_color(row: Any, col: Any, shape: Any, color: Any) -> None:
        color_at(center(row, col, shape), color)

    def points(xy: Any, colors: Any, occlusion: int = 6) -> None:
        xy = np.asarray(xy)
        for i, (point, color) in enumerate(zip(xy, colors, strict=True)):
            if not np.any(
                np.linalg.norm(xy[i + 1 :] - point, axis=-1) < occlusion
            ):
                color_at(point, color, 2)

    def rowcols(rc: Any, shape: Any) -> np.ndarray:
        return np.asarray([center(r, c, shape) for r, c in rc])

    def pos(p: dict[str, Any]) -> tuple[int, int]:
        return (int(p["row"]), int(p["col"]))

    if task_id in ("Maze-v0", "Cleaner-v0"):
        grid = state["walls"] if task_id == "Maze-v0" else state["grid"]
        if task_id == "Maze-v0":
            rc = [pos(state["target_position"]), pos(state["agent_position"])]
            colors = [(0, 210, 70), (220, 30, 30)]
            cell_colors = [white, (0, 0, 0)]
        else:
            rc = state["agents_locations"]
            colors = palette[: len(rc)]
            cell_colors = [(0, 240, 40), (250, 250, 250), (0, 0, 0)]
        xy = rowcols(rc, grid.shape)
        for r, c in np.ndindex(grid.shape):
            point = center(r, c, grid.shape)
            if np.all(np.linalg.norm(xy - point, axis=-1) > 7):
                color_at(point, cell_colors[int(grid[r, c])])
        points(xy, colors)
    elif task_id in ("FlatPack-v0", "Connector-v3"):
        grid = state["grid"]
        for r, c in np.ndindex(grid.shape):
            value = int(grid[r, c])
            index = value if task_id == "FlatPack-v0" else (value - 1) // 3
            cell_color(
                r, c, grid.shape, white if value == 0 else palette[index % 20]
            )
    elif task_id.startswith("Sudoku") or task_id in (
        "Minesweeper-v0",
        "Game2048-v1",
        "SlidingTilePuzzle-v0",
    ):
        grid = (
            state["puzzle"]
            if task_id == "SlidingTilePuzzle-v0"
            else state["board"]
        )
        for r, c in np.ndindex(grid.shape):
            x0, x1 = (
                c * width // grid.shape[1],
                (c + 1) * width // grid.shape[1],
            )
            y0, y1 = (
                r * height // grid.shape[0],
                (r + 1) * height // grid.shape[0],
            )
            patch = image[y0 + 2 : y1 - 2, x0 + 2 : x1 - 2]
            value = int(grid[r, c])
            occupied = value >= 0 if task_id.startswith("Sudoku") else value > 0
            if task_id == "Minesweeper-v0":
                has_glyph = np.any(np.min(patch, axis=-1) < 140)
            else:
                limit = 60 if task_id == "SlidingTilePuzzle-v0" else 120
                has_glyph = np.any(np.max(patch, axis=-1) < limit)
            assert bool(has_glyph) == occupied, (
                task_id,
                r,
                c,
                value,
                "glyph presence",
            )
    elif task_id == "Snake-v1":
        grid = state["body_state"]
        head = pos(state["head_position"])
        fruit = pos(state["fruit_position"])
        for r, c in np.ndindex(grid.shape):
            if (r, c) not in (head, fruit):
                cell_color(
                    r, c, grid.shape, (40, 170, 40) if grid[r, c] > 0 else white
                )
        points(
            rowcols([fruit, head], grid.shape), [(65, 180, 40), (220, 50, 50)]
        )
    elif task_id == "PacMan-v1":
        shape = state["grid"].shape
        player = state["player_locations"]
        ghosts = state["ghost_locations"][:, ::-1]
        rc = list(ghosts) + [(player["x"], player["y"])]
        colors = [
            (80, 100, 255) if state["frightened_state_time"] > 0 else palette[i]
            for i in range(len(ghosts))
        ] + [(255, 230, 45)]
        points(rowcols(rc, shape), colors)
    elif task_id == "LevelBasedForaging-v0":
        food = state["food_items"]
        agents = state["agents"]
        rc = list(food["position"][~food["eaten"]]) + list(agents["position"])
        colors = [(210, 30, 30)] * int(np.count_nonzero(~food["eaten"])) + list(
            palette[: len(agents["id"])]
        )
        points(rowcols(rc, (8, 8)), colors)
    elif task_id == "RobotWarehouse-v0":
        shelves = state["shelves"]
        agents = state["agents"]
        shape = state["grid"].shape[1:]
        rc = list(
            zip(shelves["position"]["x"], shelves["position"]["y"], strict=True)
        ) + list(
            zip(agents["position"]["x"], agents["position"]["y"], strict=True)
        )
        colors = [
            (230, 130, 40) if requested else (100, 90, 150)
            for requested in shelves["is_requested"]
        ] + [
            (180, 110, 40) if carrying else palette[i]
            for i, carrying in enumerate(agents["is_carrying"])
        ]
        points(rowcols(rc, shape), colors, 4)
    elif task_id == "Sokoban-v0":
        grid = state["fixed_grid"]
        movable = state["variable_grid"]
        for r, c in np.ndindex(grid.shape):
            if movable[r, c] == 4:
                color = (120, 60, 20)
            elif movable[r, c] == 3:
                color = (20, 180, 50)
            elif grid[r, c] == 1:
                color = (130, 54, 24)
            else:
                color = (232, 232, 232)
            color_at(center(r, c, grid.shape), color, 2)
    elif task_id == "Tetris-v0":
        grid = state["grid_padded"][:10, :10]
        for r, c in np.ndindex(grid.shape):
            point = (
                width // 5 + (c + 0.5) * (width * 4 // 5 - width // 5) / 10,
                height // 5 + (r + 0.5) * (height - 4 - height // 5) / 10,
            )
            color_at(point, palette[0] if grid[r, c] else white)
        block = max(4, width // 4 // 6)
        for r, c in np.ndindex((4, 4)):
            point = (
                width // 2 - 2 * block + (c + 0.5) * block,
                height // 18 + (r + 0.5) * block,
            )
            color_at(
                point, palette[1] if state["new_tetromino"][r, c] else white
            )
    elif task_id in ("TSP-v1", "CVRP-v1", "MultiCVRP-v0"):
        if task_id == "MultiCVRP-v0":
            xy = state["nodes"]["coordinates"]
            active = state["nodes"]["demands"] > 0
            colors = [
                (0, 0, 0) if needed or i == 0 else palette[2]
                for i, needed in enumerate(active)
            ]
            vehicle_xy = xy[state["vehicles"]["positions"]]
            xy = np.concatenate([xy, vehicle_xy])
            colors += list(palette[: len(vehicle_xy)])
            xy = xy / 10
        else:
            xy = state["coordinates"]
            visited = state["visited_mask"]
            current = int(state["position"])
            colors = [
                palette[0]
                if task_id == "TSP-v1" and i == current
                else palette[2]
                if seen and not (task_id == "CVRP-v1" and i == 0)
                else (0, 0, 0)
                for i, seen in enumerate(visited)
            ]
        points(12 + xy * np.array([width - 24, height - 24]), colors, 7)
    elif task_id == "GraphColoring-v1":
        colors = state["colors"]
        angles = np.arange(len(colors)) * 2 * np.pi / len(colors)
        xy = np.array([width // 2, height // 2]) + np.column_stack([
            np.cos(angles),
            np.sin(angles),
        ]) * (min(width, height) * 38 // 100)
        points(xy, [palette[c % 20] if c >= 0 else white for c in colors])
        current = int(state["current_node_index"])
        color_at(xy[current] + [7, 0], palette[3], 2)
    elif task_id == "MMST-v0":
        colors = [
            white if c < 0 else palette[c % 20] for c in state["node_types"]
        ]
        for i, nodes in enumerate(state["connected_nodes"]):
            for node in nodes[nodes >= 0]:
                colors[int(node)] = palette[i]
        points(
            rowcols([(i // 6, i % 6) for i in range(len(colors))], (6, 6)),
            colors,
        )
    elif task_id.startswith("RubiksCube"):
        palette_cube = [
            white,
            (255, 214, 0),
            (0, 82, 255),
            (0, 155, 72),
            (255, 88, 0),
            (183, 18, 52),
        ]
        side = min(width // 4, height // 3) - 4
        origin = ((width - 4 * side) // 2, (height - 3 * side) // 2)
        for face, (fx, fy) in enumerate([
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (1, 2),
        ]):
            for r, c in np.ndindex((3, 3)):
                point = (
                    origin[0] + (fx + (c + 0.5) / 3) * side,
                    origin[1] + (fy + (r + 0.5) / 3) * side,
                )
                color_at(point, palette_cube[int(state["cube"][face, r, c])])
    elif task_id == "SearchAndRescue-v0":
        xy = np.concatenate([
            state["targets"]["pos"],
            state["searchers"]["pos"],
        ])
        colors = [
            (70, 190, 90) if found else (230, 90, 60)
            for found in state["targets"]["found"]
        ] + [(60, 120, 230)] * len(state["searchers"]["pos"])
        points(xy * np.array([width - 1, height - 1]), colors)
    elif task_id == "Knapsack-v1":
        packed = np.flatnonzero(state["packed_items"])
        if not len(packed):
            color_at((width * 0.5, height * 0.7), (35, 135, 95))
        else:
            xy = [
                (
                    width // 4
                    + 8
                    + (i % 10) * (width * 3 // 4 - width // 4 - 16) // 10,
                    height // 5
                    + height // 5
                    + (i // 10)
                    * (height * 4 // 5 - height // 5 - height // 5 - 8)
                    // 5,
                )
                for i in packed
            ]
            points(xy, [palette[i % 20] for i in packed])
    elif task_id == "JobShop-v0":
        starts = state["scheduled_times"]
        durations = state["ops_durations"]
        machines = state["ops_machine_ids"]
        horizon = max(
            1,
            int(state["step_count"]),
            int(np.max(np.where(starts >= 0, starts + durations, 0))),
        )
        for job, op in zip(*np.where(starts >= 0), strict=True):
            color_at(
                (
                    (starts[job, op] + durations[job, op] / 2)
                    * width
                    / horizon,
                    (machines[job, op] + 0.5)
                    * height
                    / len(state["machines_job_ids"]),
                ),
                palette[job % 20],
            )
        if not np.any(starts >= 0):
            color_at((0, height // 2), (0, 0, 0))
    elif task_id == "BinPack-v2":
        # The last drawn box is not obscured by another box's edges.
        placed = np.flatnonzero(state["items_placed"])
        if not len(placed):
            color_at((width * 0.1, height * 0.9), (180, 180, 180), 2)
        else:
            i = placed[-1]
            loc = state["items_location"]
            size = state["items"]
            container = state["container"]
            p = np.array([loc[a][i] for a in "xyz"], dtype=float)
            d = np.array([size[a + "_len"][i] for a in "xyz"], dtype=float)
            extent = np.array([container[a + "2"] for a in "xyz"])
            for axis in range(3):
                corner = p + d
                corner[axis] -= d[axis] / 2
                x, y, z = corner / extent
                color_at(
                    (
                        width * (0.1 + 0.6 * x + 0.2 * y),
                        height * (0.9 - 0.6 * z - 0.2 * y),
                    ),
                    palette[i % 20],
                    2,
                )
    else:
        raise AssertionError(("Missing render coverage", task_id))


class JumanjiOfficialAlignTest(parameterized.TestCase):
    """Official oracle alignment for every registered Jumanji task."""

    @parameterized.parameters("Game2048-v1", "Snake-v1", "Tetris-v0")
    def test_native_sample_distributions(self, task_id: str) -> None:
        """Check sampled semantic state without synchronizing a native reset."""
        env = make_gymnasium(task_id, num_envs=64, num_threads=2, seed=31)
        official = _make_official_env(task_id)
        samples = []
        try:
            key = {
                "Game2048-v1": "board",
                "Snake-v1": "grid",
                "Tetris-v0": "tetromino",
            }[task_id]
            for _ in range(64):
                obs, _ = env.reset()
                samples.append(obs[key].copy())
            values = np.concatenate(samples)
            groups = []
            if task_id == "Game2048-v1":
                boards = values.reshape(-1, 16)
                self.assertTrue(np.all(np.count_nonzero(boards, axis=1) == 1))
                self.assertTrue(np.all(np.isin(boards, (0, 1, 2))))
                groups = [
                    (boards.argmax(axis=1), np.full(16, 1 / 16)),
                    (boards.max(axis=1) - 1, np.array([0.9, 0.1])),
                ]
            elif task_id == "Snake-v1":
                for channel in (1, 3):
                    positions = values[..., channel].reshape(-1, 144)
                    self.assertTrue(np.all(positions.sum(axis=1) == 1))
                    groups.append((
                        positions.argmax(axis=1),
                        np.full(144, 1 / 144),
                    ))
                self.assertTrue(np.all(groups[0][0] != groups[1][0]))
            else:
                templates = np.asarray(official.TETROMINOES_LIST[:, 0])
                matches = np.all(values[:, None] == templates, axis=(2, 3))
                self.assertTrue(np.all(matches.sum(axis=1) == 1))
                groups = [(matches.argmax(axis=1), np.full(7, 1 / 7))]
            for labels, expected in groups:
                observed = np.bincount(labels, minlength=len(expected)) / len(
                    labels
                )
                budget = 6 * np.sqrt(expected * (1 - expected) / len(labels))
                np.testing.assert_array_less(
                    np.abs(observed - expected), budget
                )
        finally:
            env.close()
            official.close()

    def test_native_rollout_and_render_content(self) -> None:
        """Check reset plus three official-driven steps for every task."""
        for task_id in jumanji_registration.jumanji_env_ids:
            with self.subTest(task_id=task_id):
                official_env = _make_official_env(task_id)
                official_state, official_timestep = official_env.reset(
                    jax.random.PRNGKey(0)
                )
                envpool_kwargs = _envpool_kwargs_from_official_state(
                    task_id, official_state, official_timestep.observation
                )
                if task_id in (
                    "SearchAndRescue-v0",
                    "Game2048-v1",
                    "Snake-v1",
                    "Tetris-v0",
                ):
                    envpool_kwargs = {}
                if task_id == "Sokoban-v0":
                    envpool_kwargs.update(
                        base_path="/tmp/envpool-missing-boxoban",
                        sokoban_level_index=0,
                    )
                env = make_gymnasium(
                    task_id,
                    num_envs=1,
                    seed=0,
                    render_mode="rgb_array",
                    render_width=_RENDER_SIZE,
                    render_height=_RENDER_SIZE,
                    **envpool_kwargs,
                )
                official_step = _native_step_reference(task_id, official_env)
                viewer = _find_viewer(official_env)
                try:
                    envpool_obs, envpool_info = env.reset()
                    official_state, official_timestep = _native_reset_reference(
                        task_id,
                        official_env,
                        official_state,
                        official_timestep,
                        envpool_obs,
                        envpool_info,
                    )
                    expected_obs = _project(
                        _tree_to_dict(official_timestep.observation),
                        _first_env_tree(envpool_obs),
                    )
                    _assert_tree_values(
                        _first_env_tree(envpool_obs),
                        expected_obs,
                        f"{task_id} reset obs",
                    )
                    self.assertEqual(envpool_info["env_id"].tolist(), [0])
                    self._assert_render_surface(
                        env, official_env, official_state, task_id, 0
                    )

                    for step in range(1, _RENDER_ALIGN_STEPS + 1):
                        official_action = _first_valid_action(
                            official_env,
                            official_timestep.observation,
                            task_id=task_id,
                            official_state=official_state,
                        )
                        (
                            envpool_obs,
                            envpool_reward,
                            envpool_terminated,
                            envpool_truncated,
                            envpool_info,
                        ) = env.step(_envpool_action(official_action))
                        official_state, official_timestep = official_step(
                            official_state,
                            official_action,
                            envpool_obs,
                            envpool_info,
                        )
                        expected_obs = _project(
                            _tree_to_dict(official_timestep.observation),
                            _first_env_tree(envpool_obs),
                        )
                        _assert_tree_values(
                            _first_env_tree(envpool_obs),
                            expected_obs,
                            f"{task_id} step {step} obs",
                        )
                        _assert_reward_values(
                            envpool_reward,
                            official_timestep.reward,
                            envpool_info,
                            task_id,
                            f"{task_id} step {step} reward",
                        )
                        np.testing.assert_array_equal(
                            np.asarray(envpool_terminated)[0],
                            bool(official_timestep.last()),
                            err_msg=f"{task_id} step {step} terminated",
                        )
                        np.testing.assert_array_equal(
                            np.asarray(envpool_truncated)[0],
                            False,
                            err_msg=f"{task_id} step {step} truncated",
                        )
                        self.assertEqual(envpool_info["env_id"].tolist(), [0])
                        self._assert_render_surface(
                            env, official_env, official_state, task_id, step
                        )
                finally:
                    env.close()
                    viewer.close()

    @parameterized.named_parameters(("seed_0", 0), ("seed_7", 7))
    def test_independent_rollouts(self, seed: int) -> None:
        """Compare varied actions through a complete episode of every task."""
        max_steps_by_task_id = {
            task_id: max_episode_steps
            for task_id, _, max_episode_steps in jumanji_registration._TASKS
        }
        for task_id in jumanji_registration.jumanji_env_ids:
            with self.subTest(task_id=task_id):
                rng = np.random.default_rng(seed + 41)
                official_env = _make_official_env(task_id)
                official_state, official_timestep = official_env.reset(
                    jax.random.PRNGKey(seed)
                )
                envpool_kwargs = _envpool_kwargs_from_official_state(
                    task_id, official_state, official_timestep.observation
                )
                if task_id in (
                    "SearchAndRescue-v0",
                    "Game2048-v1",
                    "Snake-v1",
                    "Tetris-v0",
                ):
                    envpool_kwargs = {}
                if task_id == "Sokoban-v0":
                    envpool_kwargs.update(
                        base_path="/tmp/envpool-missing-boxoban",
                        sokoban_level_index=0,
                    )
                env = make_gymnasium(
                    task_id,
                    num_envs=1,
                    seed=seed,
                    **envpool_kwargs,
                )
                official_step = _native_step_reference(task_id, official_env)
                try:
                    envpool_obs, envpool_info = env.reset()
                    official_state, official_timestep = _native_reset_reference(
                        task_id,
                        official_env,
                        official_state,
                        official_timestep,
                        envpool_obs,
                        envpool_info,
                    )
                    expected_obs = _project(
                        _tree_to_dict(official_timestep.observation),
                        _first_env_tree(envpool_obs),
                    )
                    _assert_tree_values(
                        _first_env_tree(envpool_obs),
                        expected_obs,
                        f"{task_id} long reset obs",
                    )
                    rollout_steps = max_steps_by_task_id[task_id]
                    done_step = None
                    for step in range(1, rollout_steps + 1):
                        official_action = _first_valid_action(
                            official_env,
                            official_timestep.observation,
                            task_id=task_id,
                            official_state=official_state,
                            rng=rng,
                        )
                        (
                            envpool_obs,
                            envpool_reward,
                            envpool_terminated,
                            envpool_truncated,
                            envpool_info,
                        ) = env.step(_envpool_action(official_action))
                        official_state, official_timestep = official_step(
                            official_state,
                            official_action,
                            envpool_obs,
                            envpool_info,
                        )
                        expected_obs = _project(
                            _tree_to_dict(official_timestep.observation),
                            _first_env_tree(envpool_obs),
                        )
                        _assert_tree_values(
                            _first_env_tree(envpool_obs),
                            expected_obs,
                            f"{task_id} long step {step} obs",
                        )
                        _assert_reward_values(
                            envpool_reward,
                            official_timestep.reward,
                            envpool_info,
                            task_id,
                            f"{task_id} long step {step} reward",
                        )
                        official_done = bool(
                            _as_numpy(official_timestep.last())
                        )
                        official_terminated = official_done and not np.any(
                            _as_numpy(official_timestep.discount)
                        )
                        envpool_done = bool(
                            envpool_terminated[0] or envpool_truncated[0]
                        )
                        self.assertEqual(
                            bool(envpool_terminated[0]),
                            official_terminated,
                            msg=f"{task_id} long step {step} terminated",
                        )
                        self.assertEqual(
                            bool(envpool_truncated[0]),
                            official_done and not official_terminated,
                            msg=f"{task_id} long step {step} truncated",
                        )
                        self.assertEqual(
                            envpool_info["env_id"].tolist(),
                            [0],
                            msg=f"{task_id} long step {step} env_id",
                        )
                        if envpool_done or official_done:
                            done_step = step
                            break
                    if (
                        done_step is None
                        and rollout_steps == max_steps_by_task_id[task_id]
                    ):
                        self.fail(f"{task_id} did not finish within max steps")
                finally:
                    env.close()

    def _assert_render_surface(
        self,
        env: Any,
        official_env: Any,
        official_state: Any,
        task_id: str,
        step: int,
    ) -> None:
        actual = np.asarray(env.render(env_ids=[0])[0])
        expected = _official_render(official_env, official_state)
        self.assertEqual(actual.shape, expected.shape)
        # Matplotlib and native drawing styles differ. Check the positions and
        # contents represented by the independent oracle's semantic state.
        _assert_render_content(actual, _tree_to_dict(official_state), task_id)
        self.assertEqual(actual.dtype, np.uint8)
        self.assertGreater(int(np.ptp(actual)), 0)
        self.assertGreater(int(np.ptp(expected)), 0)


if __name__ == "__main__":
    absltest.main()
