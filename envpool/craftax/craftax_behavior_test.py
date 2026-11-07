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

from collections.abc import Iterator
from functools import lru_cache
from typing import Any

import numpy as np
from absl.testing import absltest, parameterized

from envpool.craftax.oracle import (
    assert_floats,
    assert_render,
    assert_state,
    check_world,
    flatten,
    jax,
    make_oracle,
    native,
    night_renderer,
    rollout_action,
    state_fields,
    state_layout,
    symbolic_observer,
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
    return (
        jax.jit(oracle.step_env),
        symbolic_observer(oracle, classic),
        params,
        state,
    )


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
    grid = np.full(state.map.shape, blocks.SAND.value, np.int32)
    view = grid if classic else grid[level]
    view[:, 0] = view[:, -1] = view[0, :] = view[-1, :] = blocks.STONE.value
    view[y, x - 1] = blocks.CRAFTING_TABLE.value
    view[y - 1, x - 1] = blocks.FURNACE.value
    if block is not None:
        view[y, x + 1] = block
    elif act == "PLACE_PLANT":
        view[y, x + 1] = blocks.GRASS.value
    elif act == "DO":
        view[y, x + 1] = blocks.TREE.value
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
            learned_spells=[True, False]
            if act == "READ_BOOK"
            else [True, True],
        )
        if act.startswith("ENCHANT_"):
            view[y, x + 1] = blocks.ENCHANTMENT_TABLE_FIRE.value
        if act == "ENCHANT_ARMOUR":
            fields["armour_enchantments"] = [1, 1, 1, 0]
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
    """Test combat and projectiles with every random walk blocked by terrain."""
    y, x = np.asarray(state.player_position)
    grid = np.asarray(state.map).copy()
    grid[level] = fc.BlockType.SAND.value
    mob_map = np.asarray(state.mob_map).copy()
    changes: dict[str, Any] = {}
    for kind, name, offset in (
        (0, "passive_mobs", (2, -2)),
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
        row, col = positions[level, 0]
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (row + dy, col + dx) != (y, x):
                grid[level, row + dy, col + dx] = fc.BlockType.WALL.value
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
        map=grid,
        mob_map=mob_map,
        armour_enchantments=[1, 2, 1, 0],
        sword_enchantment=1,
        bow_enchantment=2,
        player_dexterity=3,
        player_strength=4,
        player_intelligence=3,
    )


def chest_loot(fields: dict[str, np.ndarray], inventory: Any) -> dict[str, Any]:
    """Check the native chest's sampled increments before using any of them."""
    ore_names = ("coal", "iron", "diamond", "sapphire", "ruby")
    loot = {
        name: fields["inventory." + name] - np.asarray(getattr(inventory, name))
        for name in ("torches", "arrows", "potions", *ore_names)
    }
    assert loot["torches"] in (0, 4, 5, 6, 7)
    assert 0 <= loot["arrows"] <= 4
    assert np.count_nonzero(loot["potions"]) <= 1
    assert ((0 <= loot["potions"]) & (loot["potions"] <= 2)).all()
    assert sum(loot[name] > 0 for name in ore_names) <= 1
    for name, maximum in zip(ore_names, (3, 2, 1, 1, 1), strict=True):
        assert 0 <= loot[name] <= maximum
    return loot


def native_trials(
    classic: bool, state: Any, action: int, count: int = 2048
) -> Iterator[tuple[dict[str, np.ndarray], float]]:
    """Repeat the same native starting conditions on one advancing stream."""
    game = native.Game(native.Params(classic), 1729)
    layout = state_layout(state, classic)
    initial = flatten(state)
    for _ in range(count):
        game.set_state(initial)
        reward = game.step(action)
        yield state_fields(game.encode_state(), layout), reward


def assert_rates(samples: Any, probabilities: Any) -> None:
    """Use a six-standard-error budget for each predeclared event rate."""
    samples, probabilities = np.asarray(samples), np.asarray(probabilities)
    budget = 6 * np.sqrt(probabilities * (1 - probabilities) / len(samples))
    np.testing.assert_array_less(
        np.abs(samples.mean(axis=0) - probabilities), budget + 1 / len(samples)
    )


class CraftaxRandomizationTest(parameterized.TestCase):
    """Check native sampled game outcomes before the oracle uses any of them."""

    @parameterized.parameters(True, False)
    def test_generated_worlds(self, classic: bool) -> None:
        """Check each generated floor and the separate world/item/ladder streams."""
        _, _, params, state = base_state(classic)
        layout = state_layout(state, classic)
        game = native.Game(native.Params(classic), 19)
        variants: dict[str, set[bytes]] = {}
        assignments = []
        for i in range(384 if not classic else 24):
            game.reset()
            fields = state_fields(game.encode_state(), layout)
            check_world(fields, classic, bool(params.always_diamond))
            for name in ("map", "item_map", "up_ladders", "down_ladders"):
                if name in fields and i < 24:
                    floors = (
                        fields[name][None, ...] if classic else fields[name]
                    )
                    for floor, value in enumerate(floors):
                        if (floor, name) in (
                            (0, "up_ladders"),
                            (8, "down_ladders"),
                            (8, "item_map"),
                        ):
                            continue
                        variants.setdefault(f"{name}[{floor}]", set()).add(
                            value.tobytes()
                        )
            if not classic:
                mapping = fields["potion_mapping"]
                assignments.append(mapping[:, None] == np.arange(6))
                game.step(fc.Action.NOOP.value)
                np.testing.assert_array_equal(
                    state_fields(game.encode_state(), layout)["potion_mapping"],
                    mapping,
                )
        for name, values in variants.items():
            self.assertGreater(len(values), 1, name)
        if not classic:
            assert_rates(assignments, np.full((6, 6), 1 / 6))

    @parameterized.parameters(True, False)
    def test_surface_resources(self, classic: bool) -> None:
        """Hold terrain geometry fixed and measure independent ore and tree draws."""
        blocks = (cc if classic else fc).BlockType
        config = native.Params(classic)
        size = config.width
        resolutions = ((size // 16, size // 16),) * 2 + (
            (size // 8, size // 2),
            (size // 4, size // 4),
        )
        rng = np.random.default_rng(17)
        config.fractal_noise_angles = [
            rng.random((y + 1) * (x + 1), dtype=np.float32).tolist()
            for y, x in resolutions
        ]
        game = native.Game(config, 19)
        maps = []
        for _ in range(128):
            game.reset()
            # Cell zero can be overwritten when the guaranteed stone draw
            # has no candidates. Only the surface is used here.
            maps.append(game.get_state()["map"][1 : size * size])
        surfaces = np.asarray(maps)
        coal, iron, diamond, stone = (
            blocks.COAL.value,
            blocks.IRON.value,
            blocks.DIAMOND.value,
            blocks.STONE.value,
        )
        eligible = np.isin(surfaces, (coal, iron, diamond, stone))
        np.testing.assert_array_equal(
            eligible, np.broadcast_to(eligible[0], eligible.shape)
        )
        candidates = surfaces[eligible]
        assert_rates(candidates == coal, 0.04 if classic else 0.03)
        candidates = candidates[candidates != coal]
        assert_rates(candidates == iron, 0.03 if classic else 0.02)
        if not classic:
            assert_rates(candidates[candidates != iron] == diamond, 0.001)

        trees = surfaces == blocks.TREE.value
        # Use separate pilot resets to find tiles with suitable tree noise.
        tree_tiles = trees[:64].any(axis=0)
        self.assertTrue(tree_tiles.any())
        assert_rates(trees[64:, tree_tiles].ravel(), 0.2)

    @parameterized.parameters(True, False)
    def test_sapling_sampling(self, classic: bool) -> None:
        """Only a successful grass interaction can yield a sapling and reward."""
        constants = cc if classic else fc
        action = constants.Action.DO.value
        state = arena(classic, action, block=constants.BlockType.GRASS.value)
        state = state.replace(inventory=replace(state.inventory, sapling=0))
        # Isolate the sapling reward from full Craftax's inventory milestones.
        achievements = np.ones_like(state.achievements)
        achievement = constants.Achievement.COLLECT_SAPLING.value
        achievements[achievement] = False
        state = replace(state, achievements=achievements)
        collected = []
        for fields, reward in native_trials(classic, state, action):
            count = fields["inventory.sapling"]
            self.assertIn(count, (0, 1))
            self.assertEqual(fields["achievements"][achievement], count)
            self.assertEqual(reward, count)
            collected.append(count == 1)
        assert_rates(collected, 0.1)

    def test_chest_sampling(self) -> None:
        """Check loot support, weighted tiers and independently awarded drops."""
        action = fc.Action.DO.value
        state = arena(False, action, level=1, block=fc.BlockType.CHEST.value)
        state = state.replace(
            inventory=replace(
                state.inventory,
                torches=0,
                arrows=0,
                potions=[0] * 6,
                coal=0,
                iron=0,
                diamond=0,
                ruby=0,
                sapphire=0,
                pickaxe=0,
                sword=0,
                bow=0,
            )
        )
        loot_samples = []
        tools = []
        for fields, _ in native_trials(False, state, action, 4096):
            loot_samples.append(chest_loot(fields, state.inventory))
            self.assertEqual(fields["inventory.bow"], 1)
            self.assertTrue(fields["chests_opened"][1])
            tiers = np.array([
                fields["inventory." + name] for name in ("pickaxe", "sword")
            ])
            self.assertTrue(((0 <= tiers) & (tiers <= 4)).all())
            self.assertLessEqual(np.count_nonzero(tiers), 1)
            tools.append(tiers[:, None] == np.arange(1, 5))

        def amounts(name: str) -> np.ndarray:
            return np.asarray([loot[name] for loot in loot_samples])

        torches = amounts("torches")
        arrows = amounts("arrows")
        potions = amounts("potions")
        ore = np.stack(
            [
                amounts(name)
                for name in ("coal", "iron", "diamond", "sapphire", "ruby")
            ],
            axis=1,
        )
        assert_rates(torches[:, None] == np.arange(4, 8), np.full(4, 0.6 / 4))
        assert_rates(arrows[:, None] == np.arange(1, 5), np.full(4, 0.25 / 4))
        assert_rates(
            potions[:, :, None] == np.arange(1, 3), np.full((6, 2), 0.5 / 12)
        )
        assert_rates(ore > 0, 0.6 * np.array([0.3, 0.3, 0.15, 0.125, 0.125]))
        assert_rates(
            tools, np.tile(0.1 * np.array([0.4, 0.3, 0.2, 0.1]), (2, 1))
        )
        # Loot groups draw independently even when the same chest awards both.
        assert_rates((torches > 0) & (arrows > 0), 0.6 * 0.25)

    def test_spell_sampling(self) -> None:
        """Learn each previously unknown spell with equal probability."""
        action = fc.Action.READ_BOOK.value
        state = replace(arena(False, action), learned_spells=[False, False])
        learned = []
        for fields, _ in native_trials(False, state, action, 1024):
            spells = fields["learned_spells"]
            self.assertEqual(spells.sum(), 1)
            self.assertEqual(fields["inventory.books"], 1)
            self.assertEqual(
                fields["achievements"][fc.Achievement.LEARN_FIREBALL.value],
                spells[0],
            )
            self.assertEqual(
                fields["achievements"][fc.Achievement.LEARN_ICEBALL.value],
                spells[1],
            )
            learned.append(spells)
        assert_rates(learned, np.full(2, 0.5))

    @parameterized.parameters((0, 2, 0, 1), (1, 2, 1, 2))
    def test_armour_sampling(
        self, first: int, second: int, third: int, fourth: int
    ) -> None:
        """Prefer empty enchantments, then pick a piece of the opposing type."""
        action = fc.Action.ENCHANT_ARMOUR.value
        before = np.array([first, second, third, fourth])
        state = replace(arena(False, action), armour_enchantments=before)
        eligible = before == (0 if np.any(before == 0) else 2)
        choices = []
        for fields, _ in native_trials(False, state, action, 1024):
            actual = fields["armour_enchantments"]
            changed = actual != before
            self.assertEqual(changed.sum(), 1)
            self.assertTrue(np.all(actual[changed] == 1))
            self.assertTrue(
                fields["achievements"][fc.Achievement.ENCHANT_ARMOUR.value]
            )
            self.assertEqual(
                fields["inventory.ruby"], int(state.inventory.ruby) - 1
            )
            choices.append(changed)
        assert_rates(choices, eligible / eligible.sum())

    @parameterized.parameters(
        (True, 0.8), (True, 0.0), (False, 0.8), (False, 0.0)
    )
    def test_spawn_sampling(self, classic: bool, daylight: float) -> None:
        """Check native spawn rates, legal distance, occupancy and mob slots."""
        constants = cc if classic else fc
        action = constants.Action.NOOP.value
        state = arena(classic, action)
        y, x = map(int, state.player_position)
        grid = np.asarray(state.map).copy()
        terrain = grid if classic else grid[0]
        positions = [(y - 10, x), (y + 10, x), (y, x - 10), (y, x + 10)]
        ranged = (
            [(y + dy, x + dx) for dy in (-5, 5) for dx in (-5, 5)]
            if classic
            else positions
        )
        for position in positions:
            terrain[position] = constants.BlockType.GRASS.value
        for position in ranged:
            terrain[position] = (
                constants.BlockType.PATH.value
                if classic
                else constants.BlockType.GRASS.value
            )
        # These two tempting locations are outside both spawn ranges.
        terrain[y, x + 3] = terrain[y, x + 14] = constants.BlockType.GRASS.value
        state = replace(state, map=grid, light_level=daylight)
        names = (
            ("cows", "zombies", "skeletons")
            if classic
            else ("passive_mobs", "melee_mobs", "ranged_mobs")
        )
        candidates = (
            positions,
            positions + ranged if classic else positions,
            ranged,
        )
        outcomes = []
        unoccupied_choices: list[list[list[bool]]] = [[], [], []]
        for fields, _ in native_trials(classic, state, action, 4096):
            row = []
            spawned: list[tuple[float, float]] = []
            for index, (name, valid) in enumerate(
                zip(names, candidates, strict=True)
            ):
                mask = fields[name + ".mask"].astype(bool)
                actual = fields[name + ".position"][mask]
                self.assertLessEqual(len(actual), 1)
                row.append(bool(len(actual)))
                if len(actual):
                    position = (actual[0, 0], actual[0, 1])
                    self.assertIn(position, valid)
                    # Earlier groups can occupy a candidate for later draws.
                    if not spawned:
                        unoccupied_choices[index].append([
                            position == cell for cell in valid
                        ])
                    spawned.append(position)
            occupied = fields["mob_map"] if classic else fields["mob_map"][0]
            self.assertEqual(np.count_nonzero(occupied), len(spawned))
            self.assertEqual(
                {tuple(p) for p in np.argwhere(occupied)}, set(spawned)
            )
            outcomes.append(row)
        probabilities = np.array([0.1, 0.02 + 0.1 * (1 - daylight) ** 2, 0.05])
        assert_rates(outcomes, probabilities)
        for choices, valid in zip(unoccupied_choices, candidates, strict=True):
            assert_rates(choices, np.full(len(valid), 1 / len(valid)))

    @parameterized.parameters(
        (True, "cows"),
        (False, "passive_mobs"),
        (True, "zombies"),
        (False, "melee_mobs"),
        (True, "skeletons"),
        (False, "ranged_mobs"),
    )
    def test_mob_movement(self, classic: bool, name: str) -> None:
        """Measure walk, chase and tied-axis choices on an unobstructed grid."""
        constants = cc if classic else fc
        # Classic activates positive-health slots on DO. Start them inactive
        # to check that the resulting walk still follows its native law.
        action = (cc.Action.DO if classic else fc.Action.NOOP).value
        state = arena(classic, action, block=constants.BlockType.SAND.value)
        y, x = map(int, state.player_position)
        passive = name in ("cows", "passive_mobs")
        melee = name in ("zombies", "melee_mobs")
        if passive:
            start = (y + 6, x)
            probabilities = (0.125, 0.125, 0.125, 0.125, 0.5)
        elif melee:
            start = (y + 4, x + 4)
            probabilities = (0.4375, 0.0625, 0.4375, 0.0625, 0.0)
        else:
            start = (y + 11, x)
            probabilities = (0.3625, 0.2125, 0.2125, 0.2125, 0.0)
        mobs = getattr(state, name)
        position = np.asarray(mobs.position).copy()
        mask = np.asarray(mobs.mask).copy()
        health = np.asarray(mobs.health).copy()
        first = (0,) if classic else (0, 0)
        position[first], mask[first] = start, not classic
        health[first] = 1
        mob_map = np.zeros_like(state.mob_map)
        if not classic:
            mob_map[0, *start] = True
        state = replace(
            state.replace(**{
                name: replace(mobs, position=position, mask=mask, health=health)
            }),
            mob_map=mob_map,
        )
        moves = np.array([(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)])
        outcomes = []
        for fields, _ in native_trials(classic, state, action):
            self.assertTrue(fields[name + ".mask"][first])
            delta = fields[name + ".position"][first] - start
            choice = (moves == delta).all(axis=1)
            self.assertEqual(choice.sum(), 1)
            outcomes.append(choice)
        assert_rates(outcomes, probabilities)

    @parameterized.parameters(True, False)
    def test_night_noise(self, classic: bool) -> None:
        """Check replayable, per-pixel uniform night noise in actual renders."""
        _, _, _, state = base_state(classic)
        state = replace(state, light_level=0.0)
        game = native.Game(native.Params(classic), 82)
        game.reset()
        seed = game.get_state()["state_rng"]
        game.reset()
        second_seed = game.get_state()["state_rng"]
        frames = []
        render = night_renderer(classic, 16)
        low, high = (
            np.asarray(render(state, 0.0)),
            np.asarray(render(state, 1.0)),
        )
        span = high - low
        visible = span.min(axis=2) > 1
        for frame_seed in (seed, second_seed):
            game.set_state(dict(flatten(state), state_rng=frame_seed))
            frame = game.pixels(16)
            np.testing.assert_array_equal(frame, game.pixels(16))
            assert_render(frame, state, classic, 16)
            fraction = ((frame[visible] - low[visible]) / span[visible])[:, :1]
            assert_floats(
                frame[visible] / 255,
                (low[visible] + fraction * span[visible]) / 255,
            )
            assert_rates(
                (fraction[:, :1] >= np.arange(4) / 4)
                & (fraction[:, :1] < (np.arange(4) + 1) / 4),
                np.full(4, 0.25),
            )
            frames.append(frame)
        self.assertFalse(np.array_equal(*frames))


class CraftaxBehaviorTest(parameterized.TestCase):
    """Drive native and official games through controlled external actions."""

    def rollout(
        self,
        classic: bool,
        state: Any,
        first: int,
        seed: int,
        *,
        combat: bool = False,
        chest: bool = False,
    ) -> None:
        """Synchronize once and compare every transition through episode end."""
        step, get_obs, params, _ = base_state(classic)
        config = native.Params(classic)
        config.max_timesteps = int(params.max_timesteps)
        if not combat:
            config.mob_despawn_distance = 1
            params = params.replace(mob_despawn_distance=1)
        game = native.Game(config, seed)
        # Exactly one synchronization, before any external actions.
        game.set_state(flatten(state))
        layout = state_layout(state, classic)
        assert_floats(game.obs(), get_obs(state))
        assert_render(game.pixels(16), state, classic, 16)
        key = jax.random.PRNGKey(seed + 1000)
        choices = np.arange(17 if classic else 43)
        if combat:
            choices = (
                np.array([cc.Action.DO.value, cc.Action.NOOP.value])
                if classic
                else np.array([
                    fc.Action[name].value
                    for name in (
                        "DO",
                        "NOOP",
                        "SLEEP",
                        "REST",
                        "SHOOT_ARROW",
                        "CAST_FIREBALL",
                        "CAST_ICEBALL",
                    )
                ])
            )
        actions = np.random.default_rng(seed).choice(choices, size=192)
        actions[0] = first
        for t, action in enumerate(actions):
            action = (
                int(action)
                if t == 0
                else rollout_action(state, int(action), classic)
            )
            key, draw = jax.random.split(key)
            before = state
            _, state, reward, done, _ = step(draw, state, action, params)
            actual_reward = game.step(action)
            encoded = game.encode_state()
            message = f"seed={seed} step={t} action={action}"
            np.testing.assert_array_equal(
                actual_reward, reward, err_msg=message
            )
            self.assertEqual(game.done(), bool(done), message)
            if chest and t == 0:
                fields = state_fields(encoded, layout)
                loot = chest_loot(fields, before.inventory)
                # This fixture starts with top-tier tools. The official step
                # computes the reward, flags, opening and guaranteed rewards.
                state = state.replace(
                    inventory=replace(
                        state.inventory,
                        **{
                            name: np.asarray(getattr(before.inventory, name))
                            + amount
                            for name, amount in loot.items()
                        },
                    )
                )
            assert_state(encoded, state, layout, message)
            assert_floats(game.obs(), get_obs(state), message)
            if t % 23 == 0 or bool(done):
                assert_render(game.pixels(16), state, classic, 16, message)
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

    def test_classic_mob_activation(self) -> None:
        """Compare Classic's DO activation with the random walk blocked."""
        action = cc.Action.DO.value
        state = arena(True, action)
        y, x = np.asarray(state.player_position) + [0, 4]
        terrain = np.array(state.map)
        terrain[y - 1 : y + 2, x - 1 : x + 2] = cc.BlockType.STONE.value
        terrain[y, x] = cc.BlockType.SAND.value
        positions = np.array(state.cows.position)
        positions[0] = [y, x]
        state = replace(state, map=terrain).replace(
            cows=replace(state.cows, position=positions)
        )
        self.rollout(True, state, action, 163, combat=True)

    @parameterized.parameters(True, False)
    def test_symbolic_mob_visibility(self, classic: bool) -> None:
        """An inactive or offscreen slot cannot erase a visible mob."""
        constants = cc if classic else fc
        _, observe, _, _ = base_state(classic)
        state = arena(classic, constants.Action.NOOP.value)
        name = "skeletons" if classic else "ranged_mobs"
        mobs = getattr(state, name)
        position, mask = np.array(mobs.position), np.array(mobs.mask)
        first, second = ((0,), (1,)) if classic else ((0, 0), (0, 1))
        game = native.Game(native.Params(classic))
        game.set_state(flatten(state))
        empty = game.obs()
        position[first] = np.asarray(state.player_position) + [0, 2]
        mask[first] = True
        state = state.replace(**{
            name: replace(mobs, position=position, mask=mask)
        })
        game.set_state(flatten(state))
        visible = game.obs()
        self.assertEqual(np.count_nonzero(visible != empty), 1)
        for active in (False, True):
            position[second] = position[first] - [
                constants.OBS_DIM[0] * active,
                0,
            ]
            mask[second] = active
            fixture = state.replace(**{
                name: replace(mobs, position=position, mask=mask)
            })
            game.set_state(flatten(fixture))
            np.testing.assert_array_equal(game.obs(), visible)
            assert_floats(visible, observe(fixture))

    @parameterized.parameters(*range(9))
    def test_floors_and_combat(self, level: int) -> None:
        """Exercise floor-specific enemy types and elemental damage."""
        state = battle(arena(False, fc.Action.DO.value, level), level)
        self.rollout(False, state, fc.Action.DO.value, 200 + level, combat=True)

    @parameterized.parameters(*range(8))
    def test_boss_waves(self, progress: int) -> None:
        """Exercise each boss stage with valid spawn positions."""
        state = arena(False, fc.Action.NOOP.value, level=8)
        grid = np.asarray(state.map).copy()
        y, x = np.asarray(state.player_position)
        # A single caged grave makes the forced boss spawn and its cooldown
        # progression independently comparable with the official step.
        grid[8, y - 3, x] = fc.BlockType.GRAVE.value
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            grid[8, y - 3 + dy, x + dx] = fc.BlockType.WALL.value
        state = replace(
            state,
            map=grid,
            boss_progress=progress,
            boss_timesteps_to_spawn_this_round=7,
        )
        self.rollout(
            False, state, fc.Action.NOOP.value, 500 + progress, combat=True
        )

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

    @parameterized.parameters(1, 3, 4)
    def test_special_chests(self, level: int) -> None:
        """The first chest independently awards the floor's bow or book."""
        action = fc.Action.DO.value
        state = arena(
            False, action, level=level, block=fc.BlockType.CHEST.value
        )
        state = state.replace(
            inventory=replace(state.inventory, bow=0, books=0)
        )
        self.rollout(False, state, action, 700 + level, chest=True)

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
        self.rollout(
            classic,
            state,
            constants.Action.DO.value,
            300 + block,
            chest=name == "CHEST",
        )


if __name__ == "__main__":
    absltest.main()
