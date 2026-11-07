# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pinned oracle helpers, imported only by tests and documentation tooling."""

import ast
import inspect
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(
        Path(os.environ.get("TEST_TMPDIR", tempfile.gettempdir()))
        / "craftax-mpl"
    ),
)

import envpool.craftax.craftax_debug as native
import jax
import numpy as np
from craftax.craftax import constants as full_constants
from craftax.craftax import renderer as full_renderer
from craftax.craftax.world_gen.world_gen_configs import ALL_SMOOTHGEN_CONFIGS
from craftax.craftax_classic import constants as classic_constants
from craftax.craftax_classic import renderer as classic_renderer
from craftax.craftax_env import make_craftax_env_from_name

jax.config.update("jax_enable_x64", False)

# Only the cache location changes. All textures and calculations come from the
# pinned source; the Bazel runfiles themselves remain read-only.
_cache = (
    Path(os.environ.get("TEST_TMPDIR", tempfile.gettempdir()))
    / "craftax-v1.6.1-textures"
)
_cache.mkdir(parents=True, exist_ok=True)
_built_cache = Path(__file__).parent / "oracle_textures"
if _built_cache.is_dir():
    _cache = _built_cache
full_constants.TEXTURE_CACHE_FILE = str(_cache / "full.bz2")
classic_constants.TEXTURE_CACHE_FILE = str(_cache / "classic.bz2")


def flatten(state: Any) -> dict[str, np.ndarray]:
    """Flatten the official dataclasses without changing values or shapes."""
    result: dict[str, np.ndarray] = {}

    def visit(value: Any, prefix: str = "") -> None:
        if hasattr(value, "__dataclass_fields__"):
            for field in value.__dataclass_fields__:
                if field != "fractal_noise_angles":
                    visit(getattr(value, field), prefix + field + ".")
        else:
            result[prefix[:-1]] = np.asarray(value)

    visit(state)
    return result


def encode(state: Any, layout: Any) -> np.ndarray:
    """Serialize official values in the native diagnostic traversal order."""
    flat = flatten(state)
    assert flat.keys() == layout.keys()
    return np.concatenate([
        flat[key].reshape(-1).astype(np.float64) for key in layout
    ])


def state_layout(state: Any, classic: bool) -> dict[str, np.ndarray]:
    """Use the native field order and the oracle's shapes and types."""
    flat = flatten(state)
    order = native.Game(native.Params(classic)).get_state()
    assert flat.keys() == order.keys()
    return {key: flat[key] for key in order}


def state_fields(
    encoded: np.ndarray, layout: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Read native diagnostic values before any oracle synchronization."""
    fields = {}
    offset = 0
    for key, example in layout.items():
        fields[key] = encoded[offset : offset + example.size].reshape(
            example.shape
        )
        offset += example.size
    assert offset == encoded.size
    return fields


def check_world(
    fields: dict[str, np.ndarray], classic: bool, always_diamond: bool
) -> None:
    """Check native terrain, ladders, lighting and potion assignments."""
    blocks = (classic_constants if classic else full_constants).BlockType
    world = fields["map"]
    assert np.isin(world, [block.value for block in blocks]).all()
    center = tuple(size // 2 for size in world.shape[-2:])
    surface = world if classic else world[0]
    assert surface[center] == blocks.GRASS.value
    if always_diamond:
        assert np.any(surface == blocks.DIAMOND.value)
    if classic:
        return

    items = full_constants.ItemType
    light = fields["light_map"]
    assert np.isin(fields["item_map"], [item.value for item in items]).all()
    assert ((0 <= light) & (light <= 1)).all()
    np.testing.assert_array_equal(
        np.sort(fields["potion_mapping"]), np.arange(6)
    )
    size = world.shape[-1]
    ladders = (fields["down_ladders"], fields["up_ladders"])
    for ladder in ladders:
        assert ((0 <= ladder) & (ladder < size)).all()
    smooth = dict(zip((0, 2, 5, 6, 7, 8), range(6), strict=True))
    for level, terrain in enumerate(world):
        down, up = (tuple(ladder[level].astype(int)) for ladder in ladders)
        if level in smooth:
            index = smooth[level]
            config = ALL_SMOOTHGEN_CONFIGS
            valid = int(config.valid_ladder[index])
            for position in (down, up):
                # A map with no eligible tiles uses the upstream cell-zero
                # fallback. Otherwise both independent draws must be legal.
                assert terrain[position] == valid or (
                    position == (0, 0) and not np.any(terrain == valid)
                ), (level, position)
            enabled = (
                bool(config.ladder_down[index]),
                bool(config.ladder_up[index]),
            )
            start = np.minimum((np.asarray(up) - 4) % size, size - 9)
            torch_center = tuple(start + 4)
            assert light[level][torch_center] == 1
            if float(config.default_light[index]) == 1:
                np.testing.assert_array_equal(light[level], 1)
        else:
            enabled = (True, True)
            assert terrain[down] == terrain[up] == blocks.PATH.value
            np.testing.assert_array_equal(light[level], 1)
            assert np.any(terrain == blocks.CHEST.value)

        floor_items = fields["item_map"][level]
        # Up is written last when both ladders select the same tile.
        for active, position, item in zip(
            enabled,
            (down, up),
            (items.LADDER_DOWN, items.LADDER_UP),
            strict=True,
        ):
            actual = np.argwhere(floor_items == item.value)
            present = active and not (
                item == items.LADDER_DOWN and enabled[1] and down == up
            )
            np.testing.assert_array_equal(
                actual, np.asarray([position] if present else []).reshape(-1, 2)
            )


def sync_world(
    state: Any,
    encoded: np.ndarray,
    layout: dict[str, np.ndarray],
    classic: bool,
    params: Any,
) -> Any:
    """Copy only the validated generated world into an official reset."""
    fields = state_fields(encoded, layout)
    check_world(fields, classic, bool(params.always_diamond))
    names = (
        ("map",)
        if classic
        else (
            "map",
            "item_map",
            "light_map",
            "up_ladders",
            "down_ladders",
            "potion_mapping",
        )
    )
    if all(angle is not None for angle in params.fractal_noise_angles):
        # Coast and lava geometry depend on the supplied angles, independently
        # of random tree and ore placement. Cell zero can hold the ore fallback.
        blocks = (classic_constants if classic else full_constants).BlockType
        actual, expected = fields["map"], np.asarray(state.map)
        if not classic:
            actual, expected = actual[0], expected[0]
        for block in (blocks.WATER, blocks.SAND, blocks.LAVA):
            np.testing.assert_array_equal(
                actual.ravel()[1:] == block.value,
                expected.ravel()[1:] == block.value,
            )
    return state.replace(**{
        key: jax.numpy.asarray(fields[key], dtype=layout[key].dtype)
        for key in names
    })


def rollout_action(state: Any, action: int, classic: bool) -> int:
    """Keep independently sampled events in the directed tests."""
    constants = classic_constants if classic else full_constants
    if not classic and action in (
        full_constants.Action.READ_BOOK.value,
        full_constants.Action.ENCHANT_ARMOUR.value,
    ):
        return constants.Action.NOOP.value
    if action == constants.Action.DO.value:
        # Classic's DO reactivates slots with positive health. Their inactive
        # walks use independent streams, so exercise activation in an arena.
        if classic and any(
            np.any((np.asarray(mobs.health) > 0) & ~np.asarray(mobs.mask))
            for mobs in (state.zombies, state.cows, state.skeletons)
        ):
            return constants.Action.NOOP.value
        y, x = np.asarray(state.player_position) + np.asarray(
            constants.DIRECTIONS[int(state.player_direction)]
        )
        terrain = np.asarray(
            state.map if classic else state.map[state.player_level]
        )
        if 0 <= y < terrain.shape[0] and 0 <= x < terrain.shape[1]:
            random_loot = {constants.BlockType.GRASS.value}
            if not classic:
                random_loot.add(full_constants.BlockType.CHEST.value)
            if terrain[y, x] in random_loot:
                return constants.Action.NOOP.value
    return action


def assert_floats(actual: Any, expected: Any, context: str = "") -> None:
    """Bound float32 roundoff without emulating JAX's fusion and vector widths."""
    # The score accumulates 22 logarithms before expm1; daylight/pixel blends
    # can fuse differently in JAX. A 1e-6 relative budget covers these float32
    # operations without platform, channel or seed-specific exceptions.
    np.testing.assert_allclose(
        actual, expected, rtol=1e-6, atol=1e-7, err_msg=context
    )


def assert_state(
    actual: np.ndarray, expected: Any, layout: Any, context: str = ""
) -> None:
    """Compare encoded state by original dtype; discrete gameplay stays exact."""
    fields = flatten(expected)
    positions = {
        name + ".position"
        for name in (
            "zombies",
            "cows",
            "skeletons",
            "melee_mobs",
            "passive_mobs",
            "ranged_mobs",
        )
    }
    shaped = state_fields(actual, {key: fields[key] for key in layout})
    for key, field in shaped.items():
        if key == "state_rng":
            continue  # Only used for independent night-render noise.
        value = fields[key]
        if key in positions:
            # Empty slots wander independently. Classic DO activation is
            # exercised separately with a controlled starting position.
            active = fields[key.removesuffix(".position") + ".mask"]
            field, value = field[active], value[active]
        if np.issubdtype(value.dtype, np.floating):
            assert_floats(field, value, context + " " + key)
        else:
            np.testing.assert_array_equal(
                field, value, err_msg=context + " " + key
            )


def assert_pixels(actual: Any, expected: Any, context: str = "") -> None:
    """Bound daylight blend roundoff and, for uint8 output, its quantization."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, context
    if actual.dtype == np.uint8:
        np.testing.assert_allclose(
            actual, expected.astype(np.uint8), rtol=0, atol=1, err_msg=context
        )
    else:
        assert_floats(actual / 255, expected / 255, context)


def make_oracle(task_id: str, max_steps: int = 257) -> tuple[Any, Any]:
    """Construct the pinned factory name with the requested episode limit."""
    env = make_craftax_env_from_name(
        task_id, auto_reset="-AutoReset-" in task_id
    )
    return env, env.default_params.replace(max_timesteps=max_steps)


def factory_names() -> tuple[str, ...]:
    """Discover names in the oracle factory so coverage cannot omit new tasks."""
    tree = ast.parse(inspect.getsource(make_craftax_env_from_name))
    return tuple(
        sorted({
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("Craftax-")
            and node.value.endswith("-v1")
        })
    )


def symbolic_observer(oracle: Any, classic: bool) -> Any:
    """Use the upstream observation with nonvisible scatter writes dropped."""
    names = (
        ("zombies", "cows", "skeletons", "arrows")
        if classic
        else (
            "melee_mobs",
            "passive_mobs",
            "ranged_mobs",
            "mob_projectiles",
            "player_projectiles",
        )
    )
    dimensions = jax.numpy.asarray(
        (classic_constants if classic else full_constants).OBS_DIM
    )

    @jax.jit
    def observe(state: Any) -> Any:
        origin = state.player_position - dimensions // 2
        groups = {}
        for name in names:
            mobs = getattr(state, name)
            position = mobs.position - origin
            visible = mobs.mask & (
                (0 <= position) & (position < dimensions)
            ).all(axis=-1)
            # The official scatter writes zeros for inactive or offscreen mobs.
            # A positive out-of-bounds index drops that write without erasing
            # a live mob. This copy is only used to compute the observation.
            groups[name] = mobs.replace(
                position=jax.numpy.where(
                    visible[..., None], mobs.position, origin + dimensions
                )
            )
        return oracle.get_obs(state.replace(**groups))

    return observe


@lru_cache(maxsize=6)
def renderer(classic: bool, tile: int = 16) -> Any:
    """Compile the unchanged official renderer at one supported tile size."""
    module = classic_renderer if classic else full_renderer
    return jax.jit(module.make_craftax_pixel_renderer(tile))


@lru_cache(maxsize=6)
def night_renderer(classic: bool, tile: int) -> Any:
    """Evaluate the endpoints of the official renderer's uniform night noise."""
    module = classic_renderer if classic else full_renderer
    render = module.make_craftax_pixel_renderer(tile)

    @jax.jit
    def render_with_noise(state: Any, noise: Any) -> Any:
        # The pinned renderer has one uniform draw. Replace it only during
        # tracing so the oracle supplies both limits without sharing samples.
        with mock.patch.object(
            jax.random,
            "uniform",
            lambda _key, shape: jax.numpy.full(shape, noise),
        ):
            return render(state)

    return render_with_noise


def assert_render(
    actual: np.ndarray, state: Any, classic: bool, tile: int, context: str = ""
) -> None:
    """Keep the pixel budget on deterministic content, including the HUD."""
    if float(state.light_level) >= 0.5 or (
        not classic and int(state.player_level) != 0
    ):
        assert_pixels(actual, renderer(classic, tile)(state), context)
        return
    render = night_renderer(classic, tile)
    lower, upper = render(state, 0.0), render(state, 1.0)
    expected = np.clip(actual, lower, upper)
    assert_pixels(actual, expected, context)


def reset_info(state: Any, classic: bool) -> dict[str, Any]:
    """Construct the official information values for a nonterminal reset."""
    from craftax.craftax.envs.common import log_achievements_to_info
    from craftax.craftax_classic.envs.common import compute_score

    info = (compute_score if classic else log_achievements_to_info)(
        state, False
    )
    return dict(info, discount=1.0)


__all__ = [
    "assert_floats",
    "assert_state",
    "assert_pixels",
    "assert_render",
    "check_world",
    "native",
    "jax",
    "full_constants",
    "classic_constants",
    "flatten",
    "encode",
    "night_renderer",
    "make_oracle",
    "factory_names",
    "renderer",
    "reset_info",
    "rollout_action",
    "state_fields",
    "state_layout",
    "sync_world",
]
