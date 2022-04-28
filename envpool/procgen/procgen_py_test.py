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

import os
import time
import random
from typing import no_type_check

import numpy as np
from absl import logging
from absl.testing import absltest

from envpool.procgen import ProcgenDMEnvPool, ProcgenEnvSpec, ProcgenGymEnvPool
from envpool.procgen.procgen_envpool import _ProcgenEnvPool, _ProcgenEnvSpec


class _ProcgenEnvPoolTest(absltest.TestCase):

  def test_dummy(self) -> None:
    # dummy assertion
    self.assertTrue(1 == 1)
    print("Passed the dummy Procgen '1 == 1' test...")

  def test_config(self) -> None:
    ref_config_keys = [
      "action", "action_num", "base_path", "batch_size", "cur_time",
      "current_level_seed", "default_action", "distribution_mode",
      "episode_done", "episodes_remaining", "fixed_asset_seed", "game_n",
      "game_name", "game_type", "grid_step", "initial_reset_complete",
      "last_reward", "last_reward_timer", "level_seed_high", "level_seed_low",
      "max_num_players", "num_envs", "num_levels", "num_threads",
      "prev_level_seed", "rand_seed", "reset_count", "seed", "start_level",
      "state_num", "thread_affinity_offset", "timeout"
    ]
    default_conf = _ProcgenEnvSpec._default_config_values
    self.assertTrue(isinstance(default_conf, tuple))
    config_keys = _ProcgenEnvSpec._config_keys
    self.assertTrue(isinstance(config_keys, list))
    self.assertEqual(len(default_conf), len(config_keys))
    self.assertEqual(sorted(config_keys), sorted(ref_config_keys))
    print("Passed the Procgen config test...")

  def test_spec(self) -> None:
    conf = _ProcgenEnvSpec._default_config_values
    env_spec = _ProcgenEnvSpec(conf)
    state_spec = env_spec._state_spec
    action_spec = env_spec._action_spec
    state_keys = env_spec._state_keys
    action_keys = env_spec._action_keys
    self.assertTrue(isinstance(state_spec, tuple))
    self.assertTrue(isinstance(action_spec, tuple))
    state_spec = dict(zip(state_keys, state_spec))
    action_spec = dict(zip(action_keys, action_spec))
    # default value of state_num is RES_W * RES_H * RGB_FACTOR = 64 * 64 * 3 = 12288
    self.assertEqual(state_spec["obs"][1][-1], 12288)
    # change conf and see if it can successfully change state_spec
    conf = dict(zip(_ProcgenEnvSpec._config_keys, conf))
    conf["state_num"] = 10086
    env_spec = ProcgenEnvSpec(tuple(conf.values()))
    state_spec = dict(zip(state_keys, env_spec._state_spec))
    self.assertEqual(state_spec["obs"][1][-1], 10086)
    print("Passed the Procgen Spec test...")

  def test_envpool(self) -> None:
    conf = dict(
      zip(_ProcgenEnvSpec._config_keys, _ProcgenEnvSpec._default_config_values)
    )
    conf["num_envs"] = num_envs = 100
    conf["batch_size"] = batch = 31
    conf["num_threads"] = os.cpu_count()
    env_spec = _ProcgenEnvSpec(tuple(conf.values()))
    env = _ProcgenEnvPool(env_spec)
    state_keys = env._state_keys
    total = 100000
    env._reset(np.arange(num_envs, dtype=np.int32))
    t = time.time()
    for i in range(total):
      state = dict(zip(state_keys, env._recv()))
      action = {
        "action": random.randint(0, 14)
      }
      env._send(tuple(action.values()))
    print("Passed the Procgen Envpool test...")

if __name__ == "__main__":
  absltest.main()
