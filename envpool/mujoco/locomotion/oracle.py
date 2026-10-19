# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pinned Composer factories used only by tests and documentation tooling."""

from pathlib import Path
from typing import Any
from unittest import mock

import mujoco
import numpy as np
from dm_control.composer import arena
from dm_control.locomotion import soccer
from dm_control.locomotion.examples import (
    basic_cmu_2019,
    basic_rodent_2020,
    cmu_2020_tracking,
)
from dm_control.locomotion.mocap import cmu_mocap_data

EXAMPLE_MODULES = (basic_cmu_2019, basic_rodent_2020, cmu_2020_tracking)
_ORACLE_ASSETS = (
    Path(__file__).absolute().parents[3] / "third_party/dmc_locomotion"
)


def make_oracle(task: str, seed: int = 0, **kwargs: Any) -> Any:
    """Construct the pinned official factory with only its assets redirected."""
    # Upstream's maze/two-touch factories additionally use NumPy's module RNG.
    # EnvPool isolates that stream per environment, with the same seed.
    np.random.seed(seed)

    def mocap_path(version: str = "2020") -> str:
        return str(_ORACLE_ASSETS / f"oracle_{version}.h5")

    # Resolve Bazel runfile links before MJCF derives relative texture paths.
    # Otherwise composer/../locomotion/... can exceed Windows MAX_PATH even
    # though the same files have short, valid paths in the external repository.
    arena_xml = str(Path(arena._ARENA_XML_PATH).resolve())
    with (
        mock.patch.object(cmu_mocap_data, "get_path_for_cmu", mocap_path),
        mock.patch.object(arena, "_ARENA_XML_PATH", arena_xml),
    ):
        if task.startswith("soccer_"):
            return soccer.load(
                team_size=kwargs.pop("team_size", 2),
                walker_type=soccer.WalkerType[
                    task.removeprefix("soccer_").upper()
                ],
                random_state=seed,
                **kwargs,
            )
        module = next(
            module for module in EXAMPLE_MODULES if hasattr(module, task)
        )
        return getattr(module, task)(random_state=seed)


def activate_oracle_context(official: Any) -> None:
    """Restore GL state after another renderer ran on the same thread."""
    # DMC caches the current context in Python, but EnvPool's C++ renderer can
    # unbind it (CGL/WGL) or replace its framebuffer. Use DMC's executor so this
    # also respects EGL's dedicated rendering thread.
    gl = official.physics.contexts.gl
    with gl.make_current() as context:
        context.call(gl._platform_make_current)
        context.call(
            mujoco.mjr_setBuffer,
            mujoco.mjtFramebuffer.mjFB_OFFSCREEN,
            official.physics.contexts.mujoco.ptr,
        )


def oracle_observations(value: Any) -> dict[str, np.ndarray]:
    """Add the public player batch axis to official observation dictionaries."""
    if isinstance(value, list):
        return {
            key: np.stack([player[key] for player in value]) for key in value[0]
        }
    return {key: np.expand_dims(item, 0) for key, item in value.items()}


def _bounds(value: Any, low: Any, high: Any) -> None:
    """Check a sampled semantic quantity against its official support."""
    value = np.asarray(value)
    assert np.all(np.isfinite(value))
    assert np.all(value >= np.asarray(low) - 1e-12)
    assert np.all(value <= np.asarray(high) + 1e-12)


def _distribution_bounds(value: Any, distribution: Any) -> None:
    if hasattr(distribution, "low"):
        _bounds(value, distribution.low, distribution.high)
    else:
        np.testing.assert_allclose(value, distribution, rtol=0, atol=1e-12)


def _corridor_layout(arena: Any, model: Any) -> Any:
    """Install actual platforms/walls after checking their geometric rules."""
    original = arena.regenerate
    gaps = hasattr(arena, "_ground_body")
    body = arena._ground_body if gaps else arena._walls_body
    ids = np.flatnonzero(
        model.geom_bodyid == model.body("ground" if gaps else "walls").id
    )

    def regenerate(random_state: Any) -> None:
        original(random_state)
        template = body.find_all("geom")[0].get_attributes()
        if gaps:
            platforms = ids[1:]
            start = model.geom_pos[platforms, 0] - model.geom_size[platforms, 0]
            end = model.geom_pos[platforms, 0] + model.geom_size[platforms, 0]
            _distribution_bounds(
                2 * model.geom_size[platforms, 0], arena._platform_length
            )
            _distribution_bounds(start[1:] - end[:-1], arena._gap_length)
            np.testing.assert_allclose(start[0], 6, rtol=0, atol=1e-12)
            assert start[-1] < arena.corridor_length
            maximum_gap = getattr(arena._gap_length, "high", arena._gap_length)
            assert end[-1] + maximum_gap >= arena.corridor_length
            _bounds(model.geom_pos[ids[0]], template["pos"], template["pos"])
            _bounds(model.geom_size[ids[0]], template["size"], template["size"])
            _bounds(
                model.geom_pos[platforms, 1:],
                template["pos"][1:],
                template["pos"][1:],
            )
            _bounds(
                model.geom_size[platforms, 1:],
                template["size"][1:],
                template["size"][1:],
            )
        else:
            _distribution_bounds(2 * model.geom_size[ids, 1], arena._wall_width)
            _distribution_bounds(
                np.diff(model.geom_pos[ids, 0]), arena._wall_gap
            )
            _distribution_bounds(
                2 * model.geom_size[ids, 2], arena._wall_height
            )
            _bounds(
                model.geom_size[ids, 0],
                template["size"][0],
                template["size"][0],
            )
            np.testing.assert_allclose(
                model.geom_pos[ids, 2],
                model.geom_size[ids, 2],
                rtol=0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                model.geom_pos[ids[0], 0],
                template["pos"][0],
                rtol=0,
                atol=1e-12,
            )
            if arena._swap_wall_side:
                np.testing.assert_array_equal(
                    np.sign(model.geom_pos[ids, 1]),
                    np.sign(template["pos"][1]) * (-1) ** np.arange(len(ids)),
                )
            np.testing.assert_allclose(
                np.abs(model.geom_pos[ids, 1]),
                arena.corridor_width / 2 - model.geom_size[ids, 1],
                rtol=0,
                atol=1e-12,
            )
        body.geom.clear()
        for index in ids:
            body.add(
                "geom",
                **(
                    template
                    | {
                        "name": model.geom(index).name,
                        "pos": model.geom_pos[index].copy(),
                        "size": model.geom_size[index].copy(),
                    }
                ),
            )

    return mock.patch.object(arena, "regenerate", regenerate)


def _maze_layout(official: Any, state: dict, model: Any) -> Any:
    """Use the native maze and goal placement, not LabMaze's seed mapping."""
    from contextlib import ExitStack
    from itertools import chain

    from dm_control import mjcf
    from labmaze.fixed_maze import FixedMazeWithRandomGoals

    task = official.task
    arena = task._maze_arena
    fixed = FixedMazeWithRandomGoals(state["maze"], state["variations"])
    assert (fixed.height, fixed.width) == (arena.maze.height, arena.maze.width)
    grid = np.asarray(fixed.entity_layer)
    assert set(np.unique(grid)) <= set("* PG")
    for token in "PG":
        assert np.count_nonzero(grid == token) == np.count_nonzero(
            np.asarray(arena.maze.entity_layer) == token
        )
    assert np.all(grid[[0, -1]] == "*") and np.all(grid[:, [0, -1]] == "*")
    reachable = set()
    todo = [tuple(np.argwhere(grid == "P")[0])]
    while todo:
        row, column = todo.pop()
        if (row, column) in reachable or grid[row, column] == "*":
            continue
        reachable.add((row, column))
        todo.extend((
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ))
    assert all(tuple(goal) in reachable for goal in np.argwhere(grid == "G"))
    stack = ExitStack()
    if len(task._target_builders) > 1:
        # The reward groups independently choose a permutation of the two
        # official target colors; retain the official texture generators.
        builders = []
        for group in range(len(task._target_builders)):
            texture = model.texture(f"target_{group}_0/target_sphere").id
            start = model.tex_adr[texture]
            size = model.tex_height[texture] * model.tex_width[texture] * 3
            pixels = model.tex_data[start : start + size].reshape(-1, 3)
            channel = int(pixels.max(axis=0).argmax())
            builders.append(
                next(
                    builder
                    for builder in task._target_builders
                    if np.asarray(builder.keywords["rgb1"]).argmax() == channel
                )
            )
        assert len(set(builders)) == len(builders)
        stack.enter_context(
            mock.patch.object(task, "_target_builders", builders)
        )
        stack.enter_context(
            mock.patch.object(task, "_shuffle_target_builders", False)
        )
    original = task.initialize_episode_mjcf

    def initialize(random_state: Any) -> None:
        original(random_state)
        positions = []
        for target in chain.from_iterable(task._active_targets):
            frame = mjcf.get_attachment_frame(target.mjcf_model)
            position = model.body(frame.full_identifier).pos.copy()
            assert np.any(
                np.all(
                    np.isclose(
                        arena.target_positions, position, rtol=0, atol=1e-12
                    ),
                    axis=-1,
                )
            )
            positions.append(tuple(position))
            frame.pos = position
        assert len(set(positions)) == len(positions)

    arena._maze = fixed
    stack.enter_context(
        mock.patch.object(task, "initialize_episode_mjcf", initialize)
    )
    return stack


def _maze_textures(official: Any, native: Any) -> None:
    """Check source textures and permitted material choices before copying."""
    model = official.physics.model.ptr
    np.testing.assert_array_equal(native.tex_data, model.tex_data)
    arena = official.task._maze_arena
    allowed = {
        "wall": {
            texture.full_identifier
            for group in arena._wall_textures.values()
            for texture in group
        },
        "floor": {texture.full_identifier for texture in arena._floor_textures},
    }
    for index in range(model.nmat):
        name = model.material(index).name
        kind = "wall" if name.startswith("wall*") else "floor"
        if name.startswith(("wall*", "floor_")):
            textures = set(native.mat_texid[index]) - {-1}
            assert len(textures) == 1
            assert {
                native.texture(texture).name for texture in textures
            } <= allowed[kind]
            model.mat_texid[index] = native.mat_texid[index]
        else:
            np.testing.assert_array_equal(
                native.mat_texid[index], model.mat_texid[index]
            )


def sync_physics(official: Any, state: dict) -> None:
    """Copy checked native inputs; keep independent task and control history."""
    for name in ("qpos", "qvel", "act"):
        getattr(official.physics.data, name)[:] = state[name]
    official.physics.data.time = state["time"]
    official.physics.forward()
    official.physics.data.qacc_warmstart[:] = state["warmstart"]


def reset_from_native(official: Any, env: Any) -> dict:
    """Validate native semantic reset state, then condition the official task."""
    from contextlib import ExitStack

    state = env._snapshot(include_model=True)
    native = mujoco.MjModel.from_binary_path(
        "native.mjb", {"native.mjb": state["model"]}
    )
    task = env.config["task_name"]
    with ExitStack() as stack:
        if task == "cmu_humanoid_go_to_target":
            position = native.site("target").pos
            _bounds(position, [-4, -4, 0], [4, 4, 0])
            stack.enter_context(
                mock.patch.object(
                    official.task, "_target_spawn_position", position[:2]
                )
            )
        elif task.endswith(("run_walls", "run_gaps")):
            stack.enter_context(_corridor_layout(official.task._arena, native))
        elif task.endswith("forage"):
            stack.enter_context(_maze_layout(official, state, native))
        elif task.startswith("soccer_"):
            arena = official.task.arena
            size = (
                -native.geom("//unnamed_geom_3").pos[0],
                -native.geom("//unnamed_geom_1").pos[1],
            )
            _bounds(size, arena._min_size, arena._max_size)
            if arena._keep_aspect_ratio:
                np.testing.assert_allclose(
                    size[0] / size[1], arena._max_size[0] / arena._max_size[1]
                )
            stack.enter_context(mock.patch.object(arena, "_min_size", size))
            stack.enter_context(mock.patch.object(arena, "_max_size", size))
        elif task == "cmu_humanoid_tracking":
            clip = state["reference_clip"]
            start = (
                state["reference_start"]
                + official.task._dataset.start_steps[clip]
            )
            assert (clip, start) in official.task._possible_starts
            stack.enter_context(
                mock.patch.object(
                    official.task, "_possible_starts", [(clip, start)]
                )
            )
            stack.enter_context(
                mock.patch.object(
                    official.task, "_start_probabilities", np.ones(1)
                )
            )
        official.reset()
    model = official.physics.model.ptr
    if task == "rodent_two_touch":
        position = native.geom("target_0_0/geom").pos
        _bounds(position, [-1.5, -1.5, 0.14], [1.5, 1.5, 0.14])
        model.geom("target_0_0/geom").pos[:] = position
    elif task == "rodent_escape_bowl":
        # Compare terrain distribution and spatial smoothness to an independent
        # official sample. Copy the actual heightfield, not random-number inputs.
        resolution = int(native.hfield_nrow[0])
        y, x = np.ogrid[-1 : 1 : resolution * 1j, -1 : 1 : resolution * 1j]
        radius = np.clip(np.sqrt(x * x + y * y), 0.1, 1)
        shape = 0.5 - np.cos(2 * np.pi * radius) / 2
        mask = shape > 0.01
        statistics = []
        for data in (native.hfield_data, model.hfield_data):
            height = data.reshape(resolution, resolution)
            np.testing.assert_allclose(height[~mask], 0, rtol=0, atol=0.02)
            smooth = np.divide(
                height, shape, out=np.zeros_like(shape), where=mask
            )
            difference = np.diff(smooth, axis=0)[mask[1:] & mask[:-1]]
            statistics.append((
                np.quantile(smooth[mask], [0.01, 0.1, 0.5, 0.9, 0.99]),
                float(np.mean(difference**2)),
            ))
        np.testing.assert_allclose(
            statistics[0][0], statistics[1][0], rtol=0, atol=0.05
        )
        _bounds(statistics[0][1] / statistics[1][1], 0.8, 1.2)
        model.hfield_data[:] = native.hfield_data
        with official.physics.contexts.gl.make_current() as context:
            context.call(
                mujoco.mjr_uploadHField,
                model,
                official.physics.contexts.mujoco.ptr,
                0,
            )
    elif task.endswith("forage"):
        _maze_textures(official, native)
    # The walker, collision geometry and actuator model remain independent.
    for name in (
        "body_mass",
        "body_inertia",
        "body_pos",
        "geom_size",
        "geom_pos",
        "jnt_range",
        "actuator_gainprm",
        "actuator_biasprm",
        "hfield_data",
        "mat_rgba",
    ):
        np.testing.assert_allclose(
            getattr(native, name),
            getattr(model, name),
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"{task}: {name}",
        )
    sync_physics(official, state)
    if task.startswith("soccer_"):
        for camera in official.task._tracking_cameras:
            camera.initialize_episode(
                official.task._tracked_entity_positions(official.physics)
            )
    return state


def _soccer_spawn(official: Any, state: dict) -> None:
    """Check actual spawn poses while keeping goal and tracker state separate."""
    sync_physics(official, state)
    task = official.task
    extent = np.asarray(task.arena.size) * task._initializer._spawn_ratio
    for entity in [task.ball, *(player.walker for player in task.players)]:
        position, quaternion = entity.get_pose(official.physics)
        _bounds(position[:2], -extent, extent)
        np.testing.assert_allclose(
            np.linalg.norm(quaternion), 1, rtol=0, atol=1e-12
        )
    position, _ = task.ball.get_pose(official.physics)
    np.testing.assert_allclose(position[2], task._initializer._init_ball_z)
    for contact in official.physics.data.contact:
        assert not any(
            task._initializer._has_relevant_contact(contact, ids)
            for ids in task._initializer._walker_geom_ids
        )


def step_from_native(
    official: Any, env: Any, action: np.ndarray
) -> tuple[Any, Any, dict]:
    """Compare successors before calling again with the next native input.

    Only actual stochastic state is copied at an independently triggered event.
    Rewards, observations, contacts, termination and task history stay official.
    """
    from contextlib import ExitStack

    sync_physics(official, env._snapshot())
    actual = env.step(action)
    state = env._snapshot()
    task = env.config["task_name"]
    throw_ins = []
    with ExitStack() as stack:
        if task == "rodent_two_touch":
            original = official.task._randomize_targets

            def targets(physics: Any) -> None:
                model = mujoco.MjModel.from_binary_path(
                    "targets.mjb",
                    {"targets.mjb": env._snapshot(include_model=True)["model"]},
                )
                position = model.geom("target_0_0/geom").pos
                _bounds(position, [-1.5, -1.5, 0.14], [1.5, 1.5, 0.14])
                previous = physics.model.geom_pos[
                    physics.model.name2id("target_0_0/geom", "geom")
                ].copy()
                assert not np.allclose(position, previous, rtol=0, atol=1e-12)
                original(physics)
                physics.named.model.geom_pos["target_0_0/geom"] = position

            stack.enter_context(
                mock.patch.object(official.task, "_randomize_targets", targets)
            )
        elif task.startswith("soccer_"):
            original_throw = official.task._throw_in
            original_spawn = official.task._initializer

            def throw_in(physics: Any, random_state: Any, ball: Any) -> None:
                before = physics.bind(ball.geom).xpos.copy()
                position = np.asarray(state["throw_in_position"])
                assert position.shape == (3,)
                _bounds(
                    position[:2],
                    np.minimum(before[:2] * 0.7, before[:2] * 0.9),
                    np.maximum(before[:2] * 0.7, before[:2] * 0.9),
                )
                np.testing.assert_allclose(position[2], 0.5)
                original_throw(physics, random_state, ball)
                ball.set_pose(physics, position)
                throw_ins.append(position)

            def spawn(task: Any, physics: Any, random_state: Any) -> None:
                original_spawn(task, physics, random_state)
                # Validate with the real initializer, not its temporary wrapper.
                with mock.patch.object(task, "_initializer", original_spawn):
                    _soccer_spawn(official, state)

            stack.enter_context(
                mock.patch.object(official.task, "_throw_in", throw_in)
            )
            stack.enter_context(
                mock.patch.object(official.task, "_initializer", spawn)
            )
        activate_oracle_context(official)
        expected = official.step(
            action if task.startswith("soccer_") else action[0]
        )
    if task.startswith("soccer_"):
        assert bool(throw_ins) == bool(state["throw_in_position"])
    return actual, expected, state
