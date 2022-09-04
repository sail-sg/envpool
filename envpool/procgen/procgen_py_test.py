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
            "state_num", "gym_reset_return_info", "timeout", "thread_affinity_offset"
        ]
        default_conf = _ProcgenEnvSpec._default_config_values
        print("The default_conf key is ")
        print(_ProcgenEnvSpec._config_keys)
        self.assertTrue(isinstance(default_conf, tuple))
        config_keys = _ProcgenEnvSpec._config_keys
        self.assertTrue(isinstance(config_keys, list))
        self.assertEqual(len(default_conf), len(config_keys))
        self.assertEqual(sorted(config_keys), sorted(ref_config_keys))
        print("Passed the Procgen config test...")


if __name__ == "__main__":
    absltest.main()