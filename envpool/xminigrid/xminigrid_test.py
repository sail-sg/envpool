# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native generation and independent rollouts against xminigrid 0.9.3."""

from collections.abc import Iterable
from contextlib import closing
from typing import Any

import jax
import numpy as np
import xminigrid
from absl.testing import absltest, parameterized
from xminigrid.core.constants import Tiles

import envpool.xminigrid.registration  # noqa: F401
from envpool.registration import make_dm, make_gymnasium, make_spec, registry
from envpool.xminigrid import RuleSet
from envpool.xminigrid.oracle import (
    RECIPE,
    oracle_ruleset,
    solution_steps,
    synchronize_reset,
)

_IDS = xminigrid.registered_environments()
_ROOM = "XLand-MiniGrid-R1-9x9"


def _walls(grid: np.ndarray, name: str) -> np.ndarray:
    """Include doors and passages when comparing static room partitions."""
    h, w = grid.shape[:2]
    walls = np.isin(grid[..., 0], [2, 8, 9])
    if name == "MiniGrid-FourRooms":
        assert np.count_nonzero(~walls[h // 2]) == 2
        assert np.count_nonzero(~walls[:, w // 2]) == 2
        walls[h // 2, :] = walls[:, w // 2] = True
    elif name.startswith("MiniGrid-DoorKey"):
        ((y, x),) = np.argwhere(grid[..., 0] == 8)
        assert 2 <= x < w - 2 and 1 <= y < w - 2
        assert walls[:, x].all()
        walls[1:-1, x] = False
    return walls


class _OracleTest(parameterized.TestCase):
    def test_registry(self) -> None:
        self.assertEqual(xminigrid.__version__, "0.9.3")
        expected = {"XMiniGrid/" + name for name in _IDS}
        expected.update(name for name in _IDS if name.startswith("XLand-"))
        self.assertSetEqual(
            expected,
            {
                name
                for name, (module, _, _) in registry.specs.items()
                if module == "envpool.xminigrid"
            },
        )
        for name in _IDS:
            ref, p = xminigrid.make(name)
            spec = make_spec("XMiniGrid/" + name)
            if name.startswith("XLand-"):
                self.assertEqual(make_spec(name).config, spec.config)
            self.assertEqual(spec.config.max_episode_steps, p.max_steps)
            self.assertEqual(
                spec.observation_space.shape, ref.observation_shape(p)
            )
            self.assertEqual(spec.observation_space.dtype, np.uint8)
            self.assertEqual(spec.action_space.n, ref.num_actions(p))

    def _check_frame(self, pool: Any, env: Any, p: Any, t: Any) -> None:
        actual = pool.render()[0]
        expected = env.render(p, jax.device_get(t))
        self.assertEqual(actual.shape, (32 * p.height, 32 * p.width, 3))
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        # Supersampling rounds at shape edges. Check each 32x32 tile so
        # a missing glyph cannot be diluted by the rest of a large board.
        diff = np.abs(actual.astype(np.int16) - expected)
        tiles = diff.reshape(p.height, 32, p.width, 32, 3)
        self.assertLessEqual(tiles.mean(axis=(1, 3)).max(), 1)

    def _rollout(
        self,
        pool: Any,
        env: Any,
        p: Any,
        initial: Any,
        obs: np.ndarray,
        info: Any,
        actions: Iterable[int] | None = None,
    ) -> float:
        name = pool.config["task"]
        np.testing.assert_array_equal(
            _walls(info["grid"][0], name),
            _walls(np.asarray(initial.state.grid), name),
        )
        t = synchronize_reset(name, initial, info)
        np.testing.assert_array_equal(obs[0], t.observation, strict=True)
        self._check_frame(pool, env, p, t)
        step = jax.jit(env.step)
        locks = info["grid"][0, ..., 0] == Tiles.DOOR_LOCKED
        moves = (
            solution_steps(pool, info)
            if actions is None
            else ((a, pool.step(np.array([a], np.int32))) for a in actions)
        )
        for i, (a, result) in enumerate(moves, 1):
            obs, reward, terminated, truncated, info = result
            t = step(p, t, np.int32(a))
            for actual, expected in (
                (obs[0], t.observation),
                (info["grid"][0], t.state.grid),
                (info["agent_pos"][0], t.state.agent.position),
                (info["direction"][0], t.state.agent.direction),
                (info["pocket"][0], t.state.agent.pocket),
                (info["elapsed_step"][0], t.state.step_num),
            ):
                np.testing.assert_array_equal(actual, expected, strict=True)
            # One float32 division and a scaled subtraction, possibly fused.
            self.assertEqual(reward.dtype, t.reward.dtype)
            self.assertAlmostEqual(
                float(reward[0]), float(t.reward), delta=2e-7
            )
            self.assertEqual(terminated[0], float(t.discount) == 0)
            self.assertEqual(truncated[0], i == p.max_steps)
            done = terminated[0] or truncated[0]
            self.assertEqual(done, int(t.step_type) == 2)
            if i == 13 or done:
                self._check_frame(pool, env, p, t)
            if done:
                if actions is None:
                    opened = info["grid"][0, ..., 0] == Tiles.DOOR_OPEN
                    self.assertTrue(opened[locks].all())
                return float(reward[0])
        raise AssertionError("Action sequence ended before the episode")

    @parameterized.parameters(*_IDS)
    def test_episode(self, name: str) -> None:
        env, p = xminigrid.make(name)
        ruleset = RECIPE if name.startswith("XLand-") else None
        if ruleset is not None:
            p = p.replace(ruleset=oracle_ruleset(ruleset))
        initial = jax.jit(env.reset)(p, jax.random.key(19))
        with closing(
            make_gymnasium(
                "XMiniGrid/" + name, ruleset=ruleset, render_mode="rgb_array"
            )
        ) as pool:
            if name.startswith("MiniGrid-Memory"):
                for correct in (False, True):
                    obs, info = pool.reset()
                    grid = info["grid"][0]
                    # Visit the cue, try picking it up, then return to the
                    # fork and choose each kind of ending.
                    x = info["agent_pos"][0, 1]
                    top = np.array_equal(grid[2, 1], grid[1, -2])
                    turn = 2 if top == correct else 1
                    actions = [1, 1] + [0] * (x - 1) + [1, 3, 1]
                    actions += [0] * (p.width - 3) + [turn, 0]
                    score = self._rollout(
                        pool, env, p, initial, obs, info, actions
                    )
                    self.assertEqual(score > 0, correct)
                return
            obs, info = pool.reset()
            if name == "MiniGrid-BlockedUnlockPickUp":
                # The blocker can cover the key or match the goal itself.
                # Choose a layout that requires clearing and opening the door.
                for _ in range(32):
                    grid = info["grid"][0]
                    ((y, x),) = np.argwhere(grid[..., 0] == Tiles.DOOR_LOCKED)
                    blocker = grid[y, x - 1]
                    if (
                        np.any(grid[..., 0] == Tiles.KEY)
                        and np.count_nonzero((grid == blocker).all(-1)) == 1
                    ):
                        break
                    obs, info = pool.reset()
                else:
                    self.fail("No layout requiring the locked door")
            if name == _ROOM or any(
                task in name for task in ("DoorKey", "LockedRoom", "Unlock")
            ):
                score = self._rollout(pool, env, p, initial, obs, info)
                self.assertGreater(score, 0)
                obs, info = pool.reset()
            actions = np.random.default_rng(42).integers(0, 6, p.max_steps)
            self._rollout(pool, env, p, initial, obs, info, actions)

    def test_outer_wall_interactions(self) -> None:
        env, base = xminigrid.make(_ROOM)
        reset = jax.jit(env.reset)
        wall, empty, ball = (2, 6), (0, 0), (3, 1)
        # Check a directed pair across the wall. When a rule consumes it,
        # step onto the edge, pick up, try to walk past it, and put down.
        probe = [4, 0, 3, 0, 4]
        base = base.replace(max_steps=base.width + len(probe))
        pool: Any
        with closing(
            make_gymnasium(
                _ROOM, max_episode_steps=base.max_steps, render_mode="rgb_array"
            )
        ) as pool:
            for d, outside in enumerate((wall, empty, empty, wall)):
                goal = (7 + d, *wall, *outside)
                consume = (4 + d, *wall, *outside, *ball)
                # Supply a key if the pickup beyond the edge found nothing.
                equip = (1, *empty, 7, 2, 0, 0)
                for rules in ([], [consume, equip]):
                    task = RuleSet(goal=goal, rules=rules)
                    pool.set_ruleset(task)
                    p = base.replace(ruleset=oracle_ruleset(task))
                    initial = reset(p, jax.random.key(3))
                    obs, info = pool.reset()
                    turns = (d - int(info["direction"][0])) % 4
                    actions = [1] * turns + [0] * (p.width - turns) + probe
                    score = self._rollout(
                        pool, env, p, initial, obs, info, actions
                    )
                    self.assertEqual(score > 0, not rules)

    def test_rule_chains_and_directions(self) -> None:
        env, base = xminigrid.make(_ROOM)
        reset = jax.jit(env.reset)
        square, star, ball = (4, 4), (12, 11), (3, 1)
        pick = (1, *square, *star, 0, 0)
        pool: Any
        with closing(make_gymnasium(_ROOM, render_mode="rgb_array")) as pool:

            def solve(
                goal: tuple[int, ...],
                rules: list[tuple[int, ...]],
                moves: list[int],
                face: int = 0,
            ) -> None:
                # Put a square at every other interior cell. Choose an
                # interior start so all compass directions can be exercised.
                task = RuleSet(goal, rules, [square] * 48)
                pool.set_ruleset(task)
                p = base.replace(ruleset=oracle_ruleset(task))
                initial = reset(p, jax.random.key(3))
                for _ in range(100):
                    obs, info = pool.reset()
                    if np.max(abs(info["agent_pos"][0] - 4)) <= 2:
                        break
                else:
                    self.fail("No interior start")
                rotate = [1] * ((face - int(info["direction"][0])) % 4)
                score = self._rollout(
                    pool, env, p, initial, obs, info, rotate + moves
                )
                self.assertGreater(score, 0)

            # Two transformations and the goal can fire on the same pickup.
            solve((1, *ball, 0, 0), [pick, (1, *star, *ball, 0, 0)], [4, 3])
            for d in range(-1, 4):
                # The undirected agent-near rule requires forward, while the
                # four directed variants also respond to put-down.
                rule, goal = (2, 3) if d == -1 else (8 + d, 11 + d)
                solve(
                    (goal, *ball, 0, 0), [(rule, *square, *ball, 0, 0)], [4, 0]
                )
                # Drop a star next to squares. The dropped cell becomes floor
                # and the product appears at every matching neighbour.
                rule, goal = (3, 4) if d == -1 else (4 + d, 7 + d)
                for first, second, left, right in (
                    (star, square, (1, 7), ball),
                    (square, star, ball, (1, 7)),
                ):
                    solve(
                        (goal, *left, *right),
                        [pick, (rule, *first, *second, *ball)],
                        [3, 4],
                        (d + 1) % 4,
                    )


class _NativeTest(parameterized.TestCase):
    def test_switching_tasks_and_time_limit(self) -> None:
        # The wall at (0, 0) satisfies the goal on the only step.
        # Stage a new ruleset only for slot 1 during play.
        task = RuleSet(goal=(5, 2, 6, 0, 0))
        pool: Any
        with closing(
            make_gymnasium(_ROOM, num_envs=2, max_episode_steps=1, ruleset=task)
        ) as pool:
            pool.reset()
            pool.set_ruleset(RuleSet(), env_id=1)
            a = np.ones(2, dtype=np.int32)
            _, reward, term, trunc, _ = pool.step(a)
            np.testing.assert_allclose(reward, 0.1, atol=1e-7)
            self.assertTrue(term.all() and trunc.all())
            for _ in range(2):
                _, _, _, _, info = pool.step(a)  # next-step autoreset
                np.testing.assert_array_equal(info["elapsed_step"], 0)
                _, _, term, trunc, _ = pool.step(a)
                np.testing.assert_array_equal(term, [True, False])
                self.assertTrue(trunc.all())

        with closing(make_dm(_ROOM, max_episode_steps=1)) as dm:
            # Recreate the first start to test an agent-position goal.
            start = dm.reset().observation.agent_pos[0]
            t = dm.step(np.array([1]))
            self.assertTrue(t.last())
            self.assertEqual(t.discount[0], 1)
        with closing(
            make_dm(_ROOM, ruleset=RuleSet(goal=(6, *start, 0, 0)))
        ) as dm:
            dm.reset()
            t = dm.step(np.array([1]))
            self.assertTrue(t.last())
            self.assertEqual(t.discount[0], 0)

    def test_invalid_task_does_not_replace_pending_task(self) -> None:
        pool: Any
        with closing(make_gymnasium(_ROOM)) as pool:
            pool.set_ruleset(RECIPE)
            for bad in (
                RuleSet(goal=(15, 0, 0, 0, 0)),
                RuleSet(goal=(6, 9, 1, 0, 0)),
                RuleSet(rules=[(12, 1, 0, 2, 0, 3, 0)]),
                RuleSet(init_tiles=[(3, 12)]),
                RuleSet(init_tiles=[(3, 2)] * 49),
            ):
                with self.assertRaises(ValueError):
                    pool.set_ruleset(bad)
            with self.assertRaises(IndexError):
                pool.set_ruleset(RuleSet(), env_id=1)
            _, info = pool.reset()
            self.assertEqual(np.count_nonzero(info["grid"][..., 0] == 5), 1)
        with self.assertRaises(ValueError):
            make_spec("XMiniGrid/MiniGrid-Unlock", ruleset=RECIPE)

    def _check_starts(
        self, name: str, starts: list[dict[str, Any]], recipe: bool
    ) -> None:
        # Axes are consecutive resets, then independently seeded slots.
        def varies(field: np.ndarray) -> None:
            self.assertTrue(
                np.any(field[1:] != field[:1]), "fixed across resets"
            )
            self.assertTrue(
                np.any(field[:, 1:] != field[:, :1]), "fixed across seeds"
            )

        grid = np.stack([s["grid"] for s in starts])
        pos = np.stack([s["agent_pos"] for s in starts])
        direction = np.stack([s["direction"] for s in starts])
        types, colors = grid[..., 0], grid[..., 1]
        batch = grid.shape[:2]
        rows, slots = np.indices(batch)
        self.assertTrue(
            np.all(types[rows, slots, pos[..., 0], pos[..., 1]] == 1)
        )
        if name.startswith("MiniGrid-Empty-"):
            self.assertTrue(np.all(pos == 1) and np.all(direction == 1))
        else:
            varies(pos)
            if "Memory" in name:
                self.assertTrue(np.all(direction == 1))
                cue, top = types[..., 2, 1], types[..., 1, -2]
                self.assertTrue(np.all(top != types[..., 5, -2]))
                for field in (cue, top, cue == top):
                    varies(field)
                return
            varies(direction)
            np.testing.assert_array_equal(np.unique(direction), np.arange(4))

        if name.startswith("XLand-") or name == "MiniGrid-Playground":
            rooms = 9 if "Playground" in name else int(name.split("-")[2][1:])
            doors = types == 9
            count = {1: 0, 2: 1, 4: 4, 6: 6, 9: 12}[rooms]
            self.assertTrue(np.all(doors.sum((-1, -2)) == count))
            if count:
                if rooms == 6:
                    self.assertTrue(np.all(doors == doors[0, 0]))
                else:
                    varies(doors)
                palette = colors[doors].reshape(*batch, count)
                varies(palette)
                np.testing.assert_array_equal(
                    np.unique(palette), np.arange(1, 7)
                )
            objects = (types > 2) & ~doors
            if recipe:
                for tile in RECIPE.init_tiles:
                    at = (grid == tile).all(-1)
                    self.assertTrue(np.all(at.sum((-1, -2)) == 1))
                    varies(at)
                self.assertTrue(np.all(objects.sum((-1, -2)) == 3))
            elif "Playground" in name:
                self.assertTrue(np.all(objects.sum((-1, -2)) == 12))
                varies(objects)
                varies(types[objects].reshape(*batch, 12))
                varies(colors[objects].reshape(*batch, 12))
            else:
                self.assertFalse(objects.any())
            return

        if "Empty" in name or "DoorKey" in name:
            self.assertTrue(np.all(grid[..., -2, -2, :] == [6, 2]))
        if name in ("MiniGrid-FourRooms", "MiniGrid-LockedRoom"):
            varies(types == 6)
        if name == "MiniGrid-FourRooms":
            varies(types == 2)
        if "DoorKey" in name or "Unlock" in name or "LockedRoom" in name:
            lock, key = types == 8, types == 7
            varies(lock)
            varies(key)
            lock_color = colors[lock].reshape(batch)
            if "DoorKey" not in name:
                varies(lock_color)
            self.assertTrue(
                np.all(~key | (colors == lock_color[..., None, None]))
            )
            if "Blocked" not in name:
                self.assertTrue(np.all(key.sum((-1, -2)) == 1))
            if "LockedRoom" in name:

                def room(mask: np.ndarray) -> np.ndarray:
                    _, _, y, x = np.nonzero(mask)
                    return y // (grid.shape[2] // 3) + 3 * (
                        x > grid.shape[3] // 2
                    )

                palette = colors[lock | (types == 9)].reshape(*batch, 6)
                self.assertTrue(np.all(np.sort(palette) == np.arange(1, 7)))
                np.testing.assert_array_equal(room(lock), room(types == 6))
                self.assertTrue(np.all(room(lock) != room(key)))
            else:
                door_x = np.nonzero(lock)[-1].reshape(batch)
                self.assertTrue(np.all(pos[..., 1] < door_x))
                i, j, _, key_x = np.nonzero(key)
                self.assertTrue(np.all(key_x < door_x[i, j]))
            if "PickUp" in name:
                mid = types.shape[-1] // 2
                objects = (types > 2) & ~lock & ~key
                sides = [slice(mid + 1, None)]
                if "Blocked" in name:
                    np.testing.assert_array_equal(
                        objects[..., mid - 1], lock[..., mid]
                    )
                    sides.append(slice(mid))
                for side in sides:
                    mask = objects[..., side]
                    self.assertTrue(np.all(mask.sum((-1, -2)) == 1))
                    varies(mask)
                    varies(types[..., side][mask].reshape(batch))
                    varies(colors[..., side][mask].reshape(batch))

    @parameterized.parameters(*_IDS)
    def test_seeded_streams(self, name: str) -> None:
        def make(threads: int) -> Any:
            return make_gymnasium(
                "XMiniGrid/" + name,
                seed=list(range(16)),
                num_envs=16,
                num_threads=threads,
                render_mode="rgb_array",
                render_height=48,
                render_width=64,
            )

        rng = np.random.default_rng(17)
        with closing(make(1)) as pool, closing(make(3)) as replay:
            for recipe in (
                [False, True] if name.startswith("XLand-") else [False]
            ):
                if recipe:
                    # Benchmarks contain padded uint8 arrays.
                    task = RuleSet(
                        np.array(RECIPE.goal, dtype=np.uint8),
                        np.array([*RECIPE.rules, [0] * 7], dtype=np.uint8),
                        np.array([*RECIPE.init_tiles, [0, 0]], dtype=np.uint8),
                    )
                    pool.set_ruleset(task)
                    replay.set_ruleset(RECIPE)
                starts = []
                for episode in range(8):
                    obs, info = pool.reset()
                    np.testing.assert_equal((obs, info), replay.reset())
                    starts.append(info)
                    if episode < 2:
                        frames = pool.render(env_ids=np.arange(16))
                        self.assertEqual(frames.shape, (16, 48, 64, 3))
                        self.assertEqual(frames.dtype, np.uint8)
                        np.testing.assert_array_equal(
                            frames[[7, 1]], replay.render(env_ids=[7, 1])
                        )
                        for act in rng.integers(
                            6, size=(24, 16), dtype=np.int32
                        ):
                            np.testing.assert_equal(
                                pool.step(act), replay.step(act)
                            )
                        np.testing.assert_array_equal(
                            pool.render(env_ids=[1, 7]),
                            replay.render(env_ids=[1, 7]),
                        )
                self._check_starts(name, starts, recipe)


if __name__ == "__main__":
    absltest.main()
