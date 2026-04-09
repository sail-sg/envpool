# Copyright 2021 Garena Online Private Limited
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
"""Unit test for dummy envpool and speed benchmark."""

import os
import time
from typing import Any

import numpy as np
from absl import logging
from absl.testing import absltest
from envpool.dummy.dummy_envpool import _DummyEnvPool, _DummyEnvSpec

from envpool.python.api import py_env
from envpool.python.protocol import EnvPool

DummyEnvSpec, _DummyDMEnvPool, _ = py_env(_DummyEnvSpec, _DummyEnvPool)


def _make_dummy_dm_env() -> EnvPool:
    config = DummyEnvSpec.gen_config(
        num_envs=2,
        batch_size=2,
        max_num_players=4,
    )
    return _DummyDMEnvPool(DummyEnvSpec(config))


def _make_multiplayer_action(
    player_count: int,
    players_env_id: np.ndarray | None = None,
) -> dict[str, object]:
    players: dict[str, np.ndarray] = {
        "id": np.arange(player_count, dtype=np.int32),
        "action": np.arange(player_count, dtype=np.int32),
    }
    action: dict[str, Any] = {
        "env_id": np.array([0, 1], dtype=np.int32),
        "list_action": np.zeros((2, 6), dtype=np.float64),
        "players": players,
    }
    if players_env_id is not None:
        players["env_id"] = players_env_id
    return action


class _DummyEnvPoolTest(absltest.TestCase):
    def test_config(self) -> None:
        ref_config_keys = [
            "num_envs",
            "batch_size",
            "num_threads",
            "max_num_players",
            "thread_affinity_offset",
            "base_path",
            "seed",
            "env_seed",
            "gym_reset_return_info",
            "state_num",
            "action_num",
            "max_episode_steps",
        ]
        default_conf = _DummyEnvSpec._default_config_values
        self.assertTrue(isinstance(default_conf, tuple))
        config_keys = _DummyEnvSpec._config_keys
        self.assertTrue(isinstance(config_keys, list))
        self.assertEqual(len(default_conf), len(config_keys))
        self.assertEqual(sorted(config_keys), sorted(ref_config_keys))

    def test_spec(self) -> None:
        conf = _DummyEnvSpec._default_config_values
        env_spec = _DummyEnvSpec(conf)
        state_spec = env_spec._state_spec
        action_spec = env_spec._action_spec
        state_keys = env_spec._state_keys
        action_keys = env_spec._action_keys
        self.assertTrue(isinstance(state_spec, tuple))
        self.assertTrue(isinstance(action_spec, tuple))
        state_spec = dict(zip(state_keys, state_spec, strict=False))
        action_spec = dict(zip(action_keys, action_spec, strict=False))
        # default value of state_num is 10
        self.assertEqual(state_spec["obs:raw"][1][-1], 10)
        self.assertEqual(state_spec["obs:dyn"][1][1][-1], 10)
        # change conf and see if it can successfully change state_spec
        # directly send dict or expose config as dict?
        conf = dict(zip(_DummyEnvSpec._config_keys, conf, strict=False))
        conf["state_num"] = 666
        env_spec = _DummyEnvSpec(tuple(conf.values()))
        state_spec = dict(zip(state_keys, env_spec._state_spec, strict=False))
        self.assertEqual(state_spec["obs:raw"][1][-1], 666)

    def test_envpool(self) -> None:
        conf = dict(
            zip(
                _DummyEnvSpec._config_keys,
                _DummyEnvSpec._default_config_values,
                strict=False,
            )
        )
        conf["num_envs"] = num_envs = 100
        conf["batch_size"] = batch = 31
        conf["num_threads"] = os.cpu_count()
        env_spec = _DummyEnvSpec(tuple(conf.values()))
        env = _DummyEnvPool(env_spec)
        state_keys = env._state_keys
        total = 100000
        env._reset(np.arange(num_envs, dtype=np.int32))
        t = time.time()
        for _ in range(total):
            state = dict(zip(state_keys, env._recv(), strict=False))
            action = {
                "env_id": state["info:env_id"],
                "players.env_id": state["info:players.env_id"],
                "list_action": np.zeros((batch, 6), dtype=np.float64),
                "players.id": state["info:players.id"],
                "players.action": state["info:players.id"],
            }
            env._send(tuple(action.values()))
        duration = time.time() - t
        fps = total * batch / duration
        logging.info(f"FPS = {fps:.6f}")

    def test_xla(self) -> None:
        conf = dict(
            zip(
                _DummyEnvSpec._config_keys,
                _DummyEnvSpec._default_config_values,
                strict=False,
            )
        )
        conf["num_envs"] = 100
        conf["batch_size"] = 31
        conf["num_threads"] = os.cpu_count()
        env_spec = _DummyEnvSpec(tuple(conf.values()))
        env = _DummyEnvPool(env_spec)
        xla_failed = False
        try:
            _ = env._xla()
        except RuntimeError:
            logging.info(
                "XLA on Dummy failed because dummy has Container typed state."
            )
            xla_failed = True
        self.assertTrue(xla_failed)

    def test_env_seed_overrides_sequential_seeding(self) -> None:
        conf = dict(
            zip(
                _DummyEnvSpec._config_keys,
                _DummyEnvSpec._default_config_values,
                strict=False,
            )
        )
        conf["num_envs"] = 3
        conf["batch_size"] = 3
        conf["max_num_players"] = 1
        conf["env_seed"] = [1, 3, 5]
        env = _DummyEnvPool(_DummyEnvSpec(tuple(conf.values())))
        env._reset(np.arange(3, dtype=np.int32))

        action = (
            np.arange(3, dtype=np.int32),
            np.arange(3, dtype=np.int32),
            np.zeros((3, 6), dtype=np.float64),
            np.zeros((3,), dtype=np.int32),
            np.zeros((3,), dtype=np.int32),
        )

        env._recv()  # consume reset output

        env._send(action)
        state = dict(zip(env._state_keys, env._recv(), strict=False))
        np.testing.assert_array_equal(
            state["done"],
            np.array([True, False, False]),
        )

        env._send(action)
        _ = env._recv()

        env._send(action)
        state = dict(zip(env._state_keys, env._recv(), strict=False))
        np.testing.assert_array_equal(
            state["done"],
            np.array([True, True, False]),
        )


class _EnvPoolMixinRegressionTest(absltest.TestCase):
    def test_from_repeats_env_id_for_uniform_multiplayer_action(self) -> None:
        env = _make_dummy_dm_env()
        action = _make_multiplayer_action(player_count=6)
        converted = env._from(action)
        np.testing.assert_array_equal(
            converted[1],
            np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        )

    def test_recv_cache_handles_variable_player_counts(self) -> None:
        env = _make_dummy_dm_env()
        env._last_players_env_id = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        action = _make_multiplayer_action(player_count=5)
        converted = env._from(action)
        np.testing.assert_array_equal(
            converted[1], np.array([0, 0, 1, 1, 1], dtype=np.int32)
        )

    def test_from_preserves_explicit_players_env_id(self) -> None:
        env = _make_dummy_dm_env()
        action = _make_multiplayer_action(
            player_count=5,
            players_env_id=np.array([0, 0, 1, 1, 1], dtype=np.int32),
        )
        converted = env._from(action)
        np.testing.assert_array_equal(
            converted[1], np.array([0, 0, 1, 1, 1], dtype=np.int32)
        )

    def test_from_raises_when_players_env_id_is_ambiguous(self) -> None:
        env = _make_dummy_dm_env()
        action = _make_multiplayer_action(player_count=5)
        with self.assertRaisesRegex(
            RuntimeError, "Cannot infer players.env_id"
        ):
            env._from(action)


if __name__ == "__main__":
    absltest.main()
