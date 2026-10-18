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
"""All-scenario behavior, randomization and oracle checks for MPE2 v1.1.1."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Any

import mpe2
import numpy as np
from absl.testing import absltest
from scipy.optimize import linear_sum_assignment

import envpool.mpe2.registration  # noqa: F401
from envpool.mpe2 import TASKS
from envpool.mpe2.oracle import (
    actions_for,
    make_oracle,
    oracle_actions,
    player_actions,
    render_frame,
    synchronize_reset,
)
from envpool.python.seed_test_utils import check_seeded_resets
from envpool.registration import make_gymnasium, make_spec, registry


def _player_order(info: dict[str, Any]) -> np.ndarray:
    return np.lexsort((info["players"]["id"], info["players"]["env_id"]))


class _MPE2Test(absltest.TestCase):
    def test_registry(self) -> None:
        upstream = {
            p.stem for p in Path(mpe2.__file__).parent.glob("*_v[0-9]*.py")
        }
        self.assertLen(upstream, 12)
        self.assertEqual(
            upstream, {alias.split("/")[1] for _, _, alias in TASKS}
        )
        for name, task_id, alias in TASKS:
            self.assertEqual(registry.specs[task_id], registry.specs[alias])
            for continuous in (False, True):
                oracle = make_oracle(name, continuous_actions=continuous)
                oracle.reset(seed=1)
                native = make_gymnasium(task_id, continuous_actions=continuous)
                try:
                    obs, info = native.reset()
                    self.assertEqual(
                        native.spec.config.max_num_players, len(oracle.agents)
                    )
                    for i, agent in enumerate(oracle.agents):
                        size = oracle.observation_space(agent).shape[0]
                        self.assertEqual(info["players"]["obs_size"][i], size)
                        self.assertTrue(np.all(obs[i, size:] == 0))
                        space = oracle.action_space(agent)
                        self.assertEqual(
                            info["players"]["action_size"][i],
                            space.shape[0] if continuous else space.n,
                        )
                        size = info["players"]["action_size"][i]
                        mask = info["players"]["action_mask"][i]
                        np.testing.assert_array_equal(mask[:size], True)
                        np.testing.assert_array_equal(mask[size:], False)
                finally:
                    native.close()
                    oracle.close()

    def _assert_state(
        self, native: Any, oracle: Any, obs: np.ndarray, info: dict[str, Any]
    ) -> None:
        raw = oracle.unwrapped
        for i, agent in enumerate(raw.possible_agents):
            expected = raw.observe(agent)
            np.testing.assert_array_equal(obs[i, : len(expected)], expected)
            np.testing.assert_array_equal(obs[i, len(expected) :], 0)
        np.testing.assert_array_equal(info["state"][0], raw.state())

    def _alignment(
        self,
        name: str,
        task_id: str,
        continuous: bool,
        seed: int,
        steps: int = 25,
        policy: Callable[[Any], np.ndarray] | None = None,
        require_success: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        native_kwargs = {**kwargs, "max_episode_steps": steps}
        oracle_kwargs = {**kwargs, "max_cycles": steps}
        native = make_gymnasium(
            task_id, continuous_actions=continuous, seed=seed, **native_kwargs
        )
        oracle = make_oracle(
            name, continuous_actions=continuous, **oracle_kwargs
        )
        try:
            oracle.reset(seed=seed)
            obs, info = native.reset()
            synchronize_reset(oracle, info)
            snapshots = [info["snapshot"][0].copy()]
            self._assert_state(native, oracle, obs, info)
            rng = np.random.default_rng(717)
            agents = oracle.possible_agents
            for step in range(steps):
                action = (
                    actions_for(native, info, rng)
                    if policy is None
                    else policy(oracle.unwrapped)
                )
                _, rew, term, trunc, _ = oracle.step(
                    oracle_actions(oracle, action)
                )
                obs, reward, terminated, truncated, info = native.step(
                    player_actions(info, action)
                )
                snapshots.append(info["snapshot"][0].copy())
                with self.subTest(
                    task=task_id, continuous=continuous, seed=seed, step=step
                ):
                    self._assert_state(native, oracle, obs, info)
                    expected = np.array([rew[a] for a in agents])
                    np.testing.assert_array_equal(
                        reward, expected.astype(np.float32)
                    )
                    np.testing.assert_array_equal(
                        info["players"]["reward"], expected
                    )
                    self.assertEqual(bool(terminated[0]), all(term.values()))
                    self.assertEqual(bool(truncated[0]), all(trunc.values()))
                    if kwargs.get("benchmark_data"):
                        for i, a in enumerate(oracle.unwrapped.world.agents):
                            value = oracle.unwrapped.scenario.benchmark_data(
                                a, oracle.unwrapped.world
                            )
                            if isinstance(value, dict):
                                self.assertEmpty(value)
                                continue
                            if name == "simple_crypto":
                                expected_info = np.concatenate(value)
                            else:
                                expected_info = np.array(
                                    value
                                    if isinstance(value, tuple)
                                    else [value],
                                    dtype=object,
                                )
                                expected_info = np.array(
                                    [
                                        -1 if x is None else x
                                        for x in expected_info
                                    ],
                                    dtype=np.float64,
                                )
                            length = info["players"]["benchmark_size"][i]
                            np.testing.assert_array_equal(
                                info["players"]["benchmark"][i, :length],
                                expected_info,
                            )
                if all(term.values()) or all(trunc.values()):
                    break
            else:
                self.fail("Episode did not reach a terminal transition")
            if require_success:
                self.assertTrue(terminated[0])
                self.assertFalse(truncated[0])
            return np.stack(snapshots)
        finally:
            native.close()
            oracle.close()

    def test_alignment(self) -> None:
        for name, task_id, _ in TASKS:
            for continuous in (False, True):
                for seed in (7, 43):
                    self._alignment(
                        name, task_id, continuous, seed, benchmark_data=True
                    )

    def test_nondefault_populations_and_neighbors(self) -> None:
        settings: dict[str, dict[str, Any]] = {
            "simple_spread": {
                "N": 5,
                "num_agent_neighbors": 7,
                "num_landmark_neighbors": 3,
                "local_ratio": 0.2,
            },
            "simple_adversary": {
                "N": 4,
                "num_agent_neighbors": 2,
                "num_landmark_neighbors": 6,
            },
            "simple_tag": {
                "num_good": 2,
                "num_adversaries": 2,
                "num_obstacles": 0,
                "num_agent_neighbors": 5,
                "num_landmark_neighbors": 2,
            },
            "simple_world_comm": {
                "num_good": 3,
                "num_adversaries": 2,
                "num_obstacles": 2,
                "num_food": 3,
                "num_forests": 3,
            },
            "collect_treasure": {
                "num_collectors": 3,
                "num_deposits": 3,
                "num_treasures": 4,
            },
            "simple_formation": {"N": 5},
            "simple_line": {"N": 1},
        }
        for name, task_id, _ in TASKS:
            if name in settings:
                for continuous in (False, True):
                    self._alignment(
                        name,
                        task_id,
                        continuous,
                        17,
                        steps=64,
                        **settings[name],
                    )

    def test_render_alignment(self) -> None:
        for name, task_id, _ in TASKS:
            for continuous, rescaling in product((False, True), repeat=2):
                native = make_gymnasium(
                    task_id,
                    seed=19,
                    continuous_actions=continuous,
                    dynamic_rescaling=rescaling,
                    render_mode="rgb_array",
                )
                oracle = make_oracle(
                    name,
                    continuous_actions=continuous,
                    dynamic_rescaling=rescaling,
                )
                try:
                    oracle.reset(seed=19)
                    _, info = native.reset()
                    synchronize_reset(oracle, info)
                    rng = np.random.default_rng(23)
                    for step in range(25):
                        if step in (0, 1, 12, 24):
                            with self.subTest(
                                task=task_id,
                                continuous=continuous,
                                rescaling=rescaling,
                                step=step,
                            ):
                                np.testing.assert_array_equal(
                                    render_frame(native), oracle.render()
                                )
                        action = actions_for(native, info, rng)
                        oracle.step(oracle_actions(oracle, action))
                        *_, info = native.step(player_actions(info, action))
                finally:
                    native.close()
                    oracle.close()

    def test_render_scales_geometry(self) -> None:
        pools = [
            make_gymnasium(
                "MPE2/Simple-v3",
                seed=29,
                render_mode="rgb_array",
                render_width=size,
                render_height=size,
            )
            for size in (350, 700)
        ]
        try:
            infos = [pool.reset()[1] for pool in pools]
            for step in range(12):
                if step in (0, 11):
                    extents = []
                    for pool in pools:
                        frame = render_frame(pool)
                        # The red landmark remains visible above the agent.
                        pixels = np.argwhere(frame[:, :, 0] > frame[:, :, 1])
                        extents.append(np.ptp(pixels, axis=0) + 1)
                    # Integer radii and outlines round separately at each size.
                    self.assertTrue(
                        np.all(np.abs(2 * extents[0] - extents[1]) <= 4)
                    )
                for i, pool in enumerate(pools):
                    action = np.array([step % 5], dtype=np.int32)
                    infos[i] = pool.step(player_actions(infos[i], action))[4]
        finally:
            for pool in pools:
                pool.close()

    def test_assignment_population_reduction(self) -> None:
        for name, task_id in (
            ("simple_line", "MPE2/SimpleLine-v1"),
            ("simple_formation", "MPE2/SimpleFormation-v1"),
        ):
            for n in (9, 17):
                for continuous in (False, True):
                    self._alignment(name, task_id, continuous, 17, N=n)

    def test_seeded_resets(self) -> None:
        for _, task_id, _ in TASKS:
            check_seeded_resets(self, task_id)

    def test_deterministic_rollouts(self) -> None:
        for _, task_id, _ in TASKS:
            for continuous in (False, True):
                a = make_gymnasium(
                    task_id,
                    num_envs=3,
                    seed=31,
                    continuous_actions=continuous,
                    render_mode="rgb_array",
                )
                b = make_gymnasium(
                    task_id,
                    num_envs=3,
                    seed=31,
                    continuous_actions=continuous,
                    render_mode="rgb_array",
                )
                try:
                    rng = np.random.default_rng(51)
                    for _ in range(2):
                        x, info = a.reset()
                        y, other_info = b.reset()
                        np.testing.assert_array_equal(
                            x[_player_order(info)], y[_player_order(other_info)]
                        )
                        for step in range(30):
                            actions = actions_for(a, info, rng)
                            other_actions = np.empty_like(actions)
                            other_actions[_player_order(other_info)] = actions[
                                _player_order(info)
                            ]
                            left = a.step(player_actions(info, actions))
                            right = b.step(
                                player_actions(other_info, other_actions)
                            )
                            info, other_info = left[4], right[4]
                            for index, (x, y) in enumerate(
                                zip(left[:4], right[:4], strict=True)
                            ):
                                left_order = (
                                    _player_order(info)
                                    if index < 2
                                    else np.argsort(info["env_id"])
                                )
                                right_order = (
                                    _player_order(other_info)
                                    if index < 2
                                    else np.argsort(other_info["env_id"])
                                )
                                np.testing.assert_array_equal(
                                    x[left_order], y[right_order]
                                )
                            if step in (0, 29):
                                np.testing.assert_array_equal(
                                    a.render([2, 0]), b.render([2, 0])
                                )
                finally:
                    a.close()
                    b.close()

    def test_invalid_configuration(self) -> None:
        for kwargs in (
            {"N": 0},
            {"N": -2},
            {"local_ratio": -0.1},
            {"num_agent_neighbors": 0},
            {"max_num_players": 2},
        ):
            with self.assertRaises(ValueError):
                make_spec("MPE2/SimpleSpread-v3", **kwargs)

    def test_independent_reset_components(self) -> None:
        """Check native semantic state, without the oracle reset synchronization."""
        for name, task_id, _ in TASKS:
            oracle = make_oracle(name)
            world = oracle.unwrapped.world
            players, entities = len(world.agents), len(world.entities)
            stride = 10 + world.dim_c

            def components(
                info: dict[str, Any],
                name: str = name,
                players: int = players,
                entities: int = entities,
                stride: int = stride,
            ) -> dict[str, np.ndarray]:
                state = info["snapshot"][np.argsort(info["env_id"])]
                records = state[:, : entities * stride].reshape(
                    4, entities, stride
                )
                goals = state[
                    :, entities * stride : entities * stride + players
                ]
                result = {
                    "agent positions": records[:, :players, :2],
                    "landmark positions": records[:, players:, :2],
                }
                if name in {
                    "simple_adversary",
                    "simple_push",
                    "simple_speaker_listener",
                    "simple_crypto",
                }:
                    result["goal"] = goals[:, 0]
                if name == "simple_reference":
                    for i in range(players):
                        result[f"agent {i} goal"] = goals[:, i]
                if name == "simple_crypto":
                    result["key"] = state[:, entities * stride + players]
                if name == "collect_treasure":
                    result["collectors"] = records[:, :6, :2]
                    result["deposits"] = records[:, 6:8, :2]
                    result["treasure types"] = records[:, players:, 9]
                if name == "simple_world_comm":
                    result["food"] = records[:, players + 1 : players + 3, :2]
                    result["forests"] = records[:, players + 3 :, :2]
                if name == "simple_line":
                    result["line direction"] = (
                        records[:, players + 1, :2] - records[:, players, :2]
                    )
                return {
                    key: value.reshape(4, -1) for key, value in result.items()
                }

            sequences = []
            for seed in (13, 13, 137):
                pool = make_gymnasium(task_id, seed=seed, num_envs=4)
                try:
                    resets = [components(pool.reset()[1]) for _ in range(24)]
                    sequences.append({
                        key: np.stack([reset[key] for reset in resets])
                        for key in resets[0]
                    })
                finally:
                    pool.close()
            oracle.close()
            for key, sequence in sequences[0].items():
                with self.subTest(task=task_id, component=key):
                    np.testing.assert_array_equal(sequence, sequences[1][key])
                    self.assertFalse(
                        np.array_equal(sequence, sequences[2][key]),
                        "Different seeds freeze this component",
                    )
                    for slot in range(4):
                        self.assertTrue(
                            np.any(sequence[:, slot] != sequence[0, slot]),
                            "Repeated resets freeze this component",
                        )
                    self.assertTrue(
                        np.any(sequence[:, 1:] != sequence[:, :1]),
                        "Parallel streams share this component",
                    )

    def test_render_selection(self) -> None:
        for _, task_id, _ in TASKS:
            pool = make_gymnasium(
                task_id,
                num_envs=3,
                seed=29,
                render_mode="rgb_array",
                render_width=350,
                render_height=350,
            )
            try:
                pool.reset()
                frames = pool.render([2, 0])
                assert frames is not None
                self.assertEqual(frames.shape, (2, 350, 350, 3))
                self.assertEqual(frames.dtype, np.uint8)
                np.testing.assert_array_equal(frames[0], render_frame(pool, 2))
                np.testing.assert_array_equal(frames[1], render_frame(pool, 0))
            finally:
                pool.close()

    def test_curriculum(self) -> None:
        for name, task_id, _ in TASKS:
            if name in {"simple_spread", "simple_tag"}:
                self._alignment(
                    name, task_id, True, 23, steps=100, curriculum=True
                )
                oracle = make_oracle(name, curriculum=True)
                try:
                    oracle.unwrapped.set_curriculum_stage(2)
                    # The inherited no-op setter in 1.1.1 keeps stage zero.
                    self.assertEqual(oracle.unwrapped.curriculum_stage, 0)
                finally:
                    oracle.close()

    @staticmethod
    def _steer(agent: Any, target: np.ndarray) -> np.ndarray:
        force = np.clip(
            2 * (target - agent.state.p_pos) - 0.8 * agent.state.p_vel, -1, 1
        )
        return np.array(
            [
                0,
                max(-force[0], 0),
                max(force[0], 0),
                max(-force[1], 0),
                max(force[1], 0),
            ],
            dtype=np.float32,
        )

    def test_success_and_time_limit_precedence(self) -> None:
        for name, task_id, _ in TASKS:
            if name not in {
                "simple_spread",
                "simple_formation",
                "simple_line",
                "simple_tag",
            }:
                continue

            def policy(raw: Any, name: str = name) -> np.ndarray:
                world = raw.world
                positions = np.array([a.state.p_pos for a in world.agents])
                if name == "simple_tag":
                    return np.stack([
                        self._steer(world.agents[0], positions[1]),
                        np.zeros(5, dtype=np.float32),
                    ])
                if name == "simple_formation":
                    angles = 0.2 + 2 * np.pi * np.arange(
                        len(world.agents)
                    ) / len(world.agents)
                    targets = world.landmarks[0].state.p_pos + 0.5 * np.stack(
                        [np.cos(angles), np.sin(angles)], axis=1
                    )
                elif name == "simple_line":
                    targets = raw.scenario._expected_positions
                else:
                    targets = np.array([
                        lm.state.p_pos for lm in world.landmarks
                    ])
                _, assignment = linear_sum_assignment(
                    np.linalg.norm(positions[:, None] - targets, axis=2)
                )
                return np.stack([
                    self._steer(agent, targets[assignment[i]])
                    for i, agent in enumerate(world.agents)
                ])

            kwargs: dict[str, Any] = (
                {"num_good": 1, "num_adversaries": 1, "num_obstacles": 0}
                if name == "simple_tag"
                else {}
            )
            with self.subTest(task=task_id):
                trajectory = self._alignment(
                    name,
                    task_id,
                    True,
                    59,
                    steps=400,
                    policy=policy,
                    require_success=True,
                    terminate_on_success=True,
                    **kwargs,
                )
                # The same final physical state must truncate, not terminate,
                # when success coincides with the configured time limit.
                self._alignment(
                    name,
                    task_id,
                    True,
                    59,
                    steps=len(trajectory) - 1,
                    policy=policy,
                    terminate_on_success=True,
                    **kwargs,
                )

    def test_treasure_pickup_delivery_and_respawn(self) -> None:
        def policy(raw: Any) -> np.ndarray:
            world = raw.world
            collector = world.agents[0]
            if collector.holding is None:
                target = min(
                    (lm for lm in world.landmarks if lm.alive),
                    key=lambda lm: np.linalg.norm(
                        lm.state.p_pos - collector.state.p_pos
                    ),
                )
            else:
                target = world.agents[1 + collector.holding]
            actions = np.zeros((4, 5), dtype=np.float32)
            actions[0] = self._steer(collector, target.state.p_pos)
            return actions

        trajectory = self._alignment(
            "collect_treasure",
            "MPE2/CollectTreasure-v1",
            True,
            41,
            steps=600,
            policy=policy,
            num_collectors=1,
            num_deposits=3,
            num_treasures=3,
        )
        # Four players and three treasures; two silent communication channels.
        entities = trajectory[:, :84].reshape(-1, 7, 12)
        holding = entities[:, 0, 8]
        self.assertGreater(
            np.count_nonzero((holding[:-1] < 0) & (holding[1:] >= 0)), 2
        )
        self.assertGreater(
            np.count_nonzero((holding[:-1] >= 0) & (holding[1:] < 0)), 2
        )
        alive = entities[:, 4:, 7].astype(bool)
        respawns = ~alive[:-1] & alive[1:]
        self.assertGreater(np.count_nonzero(respawns), 2)
        positions = entities[:, 4:, :2]
        self.assertGreater(len(np.unique(positions[1:][respawns], axis=0)), 2)
        self.assertGreater(len(np.unique(entities[1:, 4:, 9][respawns])), 1)


if __name__ == "__main__":
    absltest.main()
