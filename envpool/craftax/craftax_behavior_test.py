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
"""Directed oracle trajectories for crafting, combat, levels, plants, and items."""

from functools import lru_cache
from typing import Any

import numpy as np
from absl.testing import absltest, parameterized

from envpool.craftax.oracle import (
    assert_floats,
    assert_pixels,
    assert_state,
    flatten,
    jax,
    make_oracle,
    native,
    renderer,
)
from envpool.craftax.oracle import (
    classic_constants as cc,
)
from envpool.craftax.oracle import (
    full_constants as fc,
)

jnp = jax.numpy


def replace(state: Any, **fields: Any) -> Any:
    """Preserve each official field dtype when constructing an initial fixture."""
    return state.replace(**{
        name: jnp.asarray(value, dtype=np.asarray(getattr(state, name)).dtype)
        for name, value in fields.items()
    })


@lru_cache(maxsize=2)
def base_state(classic: bool) -> tuple[Any, Any, Any, Any]:
    """Cache the official reset and compiled functions across fixtures."""
    oracle, params = make_oracle(
        "Craftax-" + ("Classic-" if classic else "") + "Symbolic-v1", 192
    )
    _, state = oracle.reset(jax.random.PRNGKey(11), params)
    # A fresh bound method passed to jit loses its cache when each rollout
    # ends. Keep both wrappers alive so every action reuses the same compile.
    return jax.jit(oracle.step_env), jax.jit(oracle.get_obs), params, state


def arena(
    classic: bool, action: int, level: int = 0, block: int | None = None
) -> Any:
    """Prepare materials and geometry for an action before synchronization."""
    _, _, _, state = base_state(classic)
    constants = cc if classic else fc
    blocks = constants.BlockType
    act = constants.Action(action).name
    size = state.map.shape[-1]
    y, x = size // 2, size // 2
    grid = np.full(state.map.shape, blocks.GRASS.value, np.int32)
    view = grid if classic else grid[level]
    view[:, 0] = view[:, -1] = view[0, :] = view[-1, :] = blocks.STONE.value
    view[y, x - 1] = blocks.CRAFTING_TABLE.value
    view[y - 1, x - 1] = blocks.FURNACE.value
    view[y, x + 1] = blocks.GRASS.value if block is None else block
    inv = state.inventory
    resources: dict[str, Any] = {
        name: np.full_like(np.asarray(getattr(inv, name)), 8 if classic else 20)
        for name in inv.__dataclass_fields__
    }
    if classic:
        for name in (
            "wood_pickaxe",
            "stone_pickaxe",
            "iron_pickaxe",
            "wood_sword",
            "stone_sword",
            "iron_sword",
        ):
            resources[name] = 0 if act.startswith("MAKE_") else 1
    else:
        resources.update(
            pickaxe=0 if "PICKAXE" in act and act.startswith("MAKE_") else 4,
            sword=0 if "SWORD" in act and act.startswith("MAKE_") else 4,
            bow=1,
            armour=[0, 1, 2, 0],
            books=2,
            potions=[1, 2, 3, 4, 5, 6],
        )
    state = state.replace(inventory=replace(inv, **resources))
    fields: dict[str, Any] = dict(
        map=grid,
        player_position=[y, x],
        player_direction=2,
        player_health=4 if act == "REST" else 9,
        player_energy=4 if act == "SLEEP" else 9,
        player_food=5,
        player_drink=5,
    )
    if not classic:
        fields.update(
            player_level=level,
            light_map=np.full(grid.shape, 0.25, np.float32),
            item_map=np.zeros_like(grid),
            player_xp=4,
            learned_spells=[False, False]
            if act == "READ_BOOK"
            else [True, True],
        )
        if act.startswith("ENCHANT_"):
            view[y, x + 1] = blocks.ENCHANTMENT_TABLE_FIRE.value
        if act in ("ASCEND", "DESCEND"):
            fields["item_map"][level, y, x] = (
                fc.ItemType.LADDER_UP.value
                if act == "ASCEND"
                else fc.ItemType.LADDER_DOWN.value
            )
            fields["monsters_killed"] = np.full(9, 10, np.int32)
            fields["up_ladders"] = np.tile([y, x], (9, 1))
            fields["down_ladders"] = np.tile([y, x], (9, 1))
        if level == 8:
            fields.update(boss_timesteps_to_spawn_this_round=0)
    state = replace(state, **fields)
    plants = np.asarray(state.growing_plants_positions).copy()
    plants[0] = [y + 2, x]
    age = np.asarray(state.growing_plants_age).copy()
    age[0] = 587
    mask = np.asarray(state.growing_plants_mask).copy()
    mask[0] = True
    return replace(
        state,
        growing_plants_positions=plants,
        growing_plants_age=age,
        growing_plants_mask=mask,
    )


def battle(state: Any, level: int) -> Any:
    """Populate each floor with its mobs, armor, and both projectile groups."""
    y, x = np.asarray(state.player_position)
    mob_map = np.asarray(state.mob_map).copy()
    changes: dict[str, Any] = {}
    for kind, name, offset in (
        (0, "passive_mobs", (1, 1)),
        (1, "melee_mobs", (0, 1)),
        (2, "ranged_mobs", (0, 4)),
    ):
        mobs = getattr(state, name)
        positions = np.asarray(mobs.position).copy()
        health = np.asarray(mobs.health).copy()
        mask = np.asarray(mobs.mask).copy()
        types = np.asarray(mobs.type_id).copy()
        type_id = int(fc.FLOOR_MOB_MAPPING[level, kind])
        positions[level, 0] = [y + offset[0], x + offset[1]]
        health[level, 0] = fc.MOB_TYPE_HEALTH_MAPPING[type_id, kind]
        mask[level, 0] = True
        types[level, 0] = type_id
        mob_map[level, *positions[level, 0]] = True
        changes[name] = replace(
            mobs, position=positions, health=health, mask=mask, type_id=types
        )
    for player in (False, True):
        name = "player_projectiles" if player else "mob_projectiles"
        direction = (
            "player_projectile_directions"
            if player
            else "mob_projectile_directions"
        )
        mobs = getattr(state, name)
        positions = np.asarray(mobs.position).copy()
        positions[level, 0] = [y, x - 3 if player else x + 2]
        mask = np.asarray(mobs.mask).copy()
        mask[level, 0] = True
        types = np.asarray(mobs.type_id).copy()
        types[level, 0] = level % 8
        directions = np.asarray(getattr(state, direction)).copy()
        directions[level, 0] = [0, 1 if player else -1]
        changes[name] = replace(
            mobs, position=positions, mask=mask, type_id=types
        )
        changes[direction] = jnp.asarray(directions)
    state = state.replace(**changes)
    return replace(
        state,
        mob_map=mob_map,
        armour_enchantments=[1, 2, 1, 0],
        sword_enchantment=1,
        bow_enchantment=2,
        player_dexterity=3,
        player_strength=4,
        player_intelligence=3,
    )


class CraftaxBehaviorTest(parameterized.TestCase):
    """Drive native and official games only through external actions."""

    @parameterized.parameters(7, 22, 2304, 4096)
    def test_weighted_random_choices(self, size: int) -> None:
        """Check weighted sampling probabilities without a prescribed stream."""
        rng = np.random.default_rng(14)
        keys = rng.integers(0, 2**32, (16384, 2), dtype=np.uint32)
        weights = rng.random(size).astype(np.float32)
        groups = np.arange(size) * min(8, size) // size
        for values in (weights, (weights < 0.35).astype(np.float32)):
            probabilities = values / values.sum(dtype=np.float64)
            distribution = values.tolist()
            actual = np.array([
                native.choice(key.tolist(), distribution) for key in keys
            ])
            self.assertTrue(np.all((actual >= 0) & (actual < size)))
            self.assertTrue(np.all(values[actual] > 0))
            expected = np.bincount(groups, weights=probabilities)
            observed = np.bincount(groups[actual], minlength=len(expected))
            # Six standard errors make this a stable distribution check while
            # rejecting uniform sampling or frozen choices on weighted maps.
            budget = 6 * np.sqrt(expected * (1 - expected) / len(keys))
            np.testing.assert_array_less(
                np.abs(observed / len(keys) - expected), budget + 1 / len(keys)
            )

    def rollout(self, classic: bool, state: Any, first: int, seed: int) -> None:
        """Synchronize once and compare every transition through episode end."""
        step, get_obs, params, _ = base_state(classic)
        render = renderer(classic)
        config = native.Params(classic)
        config.max_timesteps = int(params.max_timesteps)
        game = native.Game(config)
        # Exactly one synchronization, before any external actions.
        game.set_state(flatten(state))
        layout = game.get_state()
        assert_floats(game.obs(), get_obs(state))
        assert_pixels(game.pixels(16), render(state))
        key = jax.random.PRNGKey(seed)
        actions = np.random.default_rng(seed).integers(
            17 if classic else 43, size=192
        )
        actions[0] = first
        for t, action in enumerate(actions):
            key, draw = jax.random.split(key)
            obs, state, reward, done, _ = step(draw, state, int(action), params)
            actual_reward = game.step(np.asarray(draw).tolist(), int(action))
            message = f"seed={seed} step={t} action={action}"
            assert_state(game.encode_state(), state, layout, message)
            assert_floats(game.obs(), obs, message)
            np.testing.assert_array_equal(
                actual_reward, reward, err_msg=message
            )
            self.assertEqual(game.done(), bool(done), message)
            if t % 23 == 0 or bool(done):
                assert_pixels(game.pixels(16), render(state), message)
            if bool(done):
                break
        self.assertTrue(
            bool(done),
            "each directed trajectory must reach an episode boundary",
        )

    @parameterized.named_parameters(
        *[("classic_" + a.name.lower(), True, a.value) for a in cc.Action],
        *[("full_" + a.name.lower(), False, a.value) for a in fc.Action],
    )
    def test_actions(self, classic: bool, action: int) -> None:
        """Exercise each action with its materials and equipment available."""
        state = arena(classic, action)
        self.rollout(classic, state, action, 100 + action)

    @parameterized.parameters(*range(9))
    def test_floors_and_combat(self, level: int) -> None:
        """Exercise floor-specific enemy types and elemental damage."""
        state = battle(arena(False, fc.Action.DO.value, level), level)
        self.rollout(False, state, fc.Action.DO.value, 200 + level)

    @parameterized.parameters(*range(8))
    def test_boss_waves(self, progress: int) -> None:
        """Exercise each boss stage with valid spawn positions."""
        state = arena(False, fc.Action.NOOP.value, level=8)
        grid = np.asarray(state.map).copy()
        y, x = np.asarray(state.player_position)
        for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            grid[8, y + dy, x + dx] = fc.BlockType.GRAVE.value
        state = replace(
            state,
            map=grid,
            boss_progress=progress,
            boss_timesteps_to_spawn_this_round=7,
        )
        self.rollout(False, state, fc.Action.NOOP.value, 500 + progress)

    def test_boss_victory(self) -> None:
        """Check the final boss hit, achievements, and terminal reward."""
        state = arena(
            False,
            fc.Action.DO.value,
            level=8,
            block=fc.BlockType.NECROMANCER.value,
        )
        state = replace(
            state, boss_progress=7, boss_timesteps_to_spawn_this_round=0
        )
        self.rollout(False, state, fc.Action.DO.value, 600)

    @parameterized.named_parameters(
        *[
            ("classic_" + name.lower(), True, name)
            for name in (
                "TREE",
                "STONE",
                "COAL",
                "IRON",
                "DIAMOND",
                "WATER",
                "RIPE_PLANT",
            )
        ],
        *[
            ("full_" + name.lower(), False, name)
            for name in (
                "TREE",
                "FIRE_TREE",
                "ICE_SHRUB",
                "STONE",
                "COAL",
                "IRON",
                "DIAMOND",
                "SAPPHIRE",
                "RUBY",
                "STALAGMITE",
                "WATER",
                "FOUNTAIN",
                "RIPE_PLANT",
                "CHEST",
            )
        ],
    )
    def test_resources(self, classic: bool, name: str) -> None:
        """Exercise extraction, food, water, and randomized chest contents."""
        constants = cc if classic else fc
        block = constants.BlockType[name].value
        state = arena(classic, constants.Action.DO.value, block=block)
        self.rollout(classic, state, constants.Action.DO.value, 300 + block)


if __name__ == "__main__":
    absltest.main()
