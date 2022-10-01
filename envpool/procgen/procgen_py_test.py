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
from typing import Any, Dict, no_type_check

import cv2
import numpy as np
from absl import logging
from absl.testing import absltest

from envpool.procgen import ProcgenDMEnvPool, ProcgenEnvSpec, ProcgenGymEnvPool
from envpool.procgen.procgen_envpool import _ProcgenEnvPool, _ProcgenEnvSpec

RES_W = 64
RES_H = 64
RGB_FACTOR = 3
ACTION_LOW = 0
ACTION_HIGH = 15
pic_count = 0

# in total 16 games in Procgen
procgen_games_list = [
  "bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun",
  "dodgeball", "fruitbot", "heist", "jumper", "leaper", "maze", "miner",
  "ninja", "plunder", "starpilot"
]


def rgb_to_picture(state: Dict):
  # convert a state's rgb 64x64x3 game observation into picture by cv2
  # for sanity check if the game is running correctly
  # state is ordered in y -> x -> rgb in one dimension array
  global pic_count
  os.makedirs("img", exist_ok=True)
  pixels = state["obs:obs"][0]
  index = 0
  img = np.zeros((RES_W, RES_H, RGB_FACTOR), np.uint8)
  for h in range(RES_W):
    for w in range(RES_H):
      for rgb in range(RGB_FACTOR):
        img[h][w][rgb] = pixels[index]
        index += 1
  pic_path = f"img/procgen_{pic_count}.png"
  pic_count += 1  # global variable increment
  cv2.imwrite(pic_path, img)


class _ProcgenEnvPoolTest(absltest.TestCase):

  def test_config(self) -> None:
    # test the config key is same as what we expect
    ref_config_keys = [
      "action", "action_num", "base_path", "batch_size", "cur_time",
      "current_level_seed", "default_action", "distribution_mode",
      "episode_done", "episodes_remaining", "fixed_asset_seed", "game_name",
      "game_type", "grid_step", "initial_reset_complete", "last_reward",
      "last_reward_timer", "level_seed_high", "level_seed_low",
      "max_num_players", "num_envs", "num_levels", "num_threads",
      "prev_level_seed", "reset_count", "seed", "start_level", "state_num",
      "gym_reset_return_info", "timeout", "thread_affinity_offset"
    ]
    default_conf = _ProcgenEnvSpec._default_config_values
    self.assertTrue(isinstance(default_conf, tuple))
    config_keys = _ProcgenEnvSpec._config_keys
    self.assertTrue(isinstance(config_keys, list))
    self.assertEqual(len(default_conf), len(config_keys))
    self.assertEqual(sorted(config_keys), sorted(ref_config_keys))

  def test_raw_envpool(self) -> None:
    # refer to : https://github.com/sail-sg/envpool/blob/main/envpool/atari/atari_envpool_test.py#L33
    # create raw procgen environment and run
    conf = dict(
      zip(
        _ProcgenEnvSpec._config_keys, _ProcgenEnvSpec._default_config_values
      )
    )
    conf["num_envs"] = num_envs = 1
    conf["batch_size"] = batch = 1
    conf["num_threads"] = os.cpu_count()
    env_spec = _ProcgenEnvSpec(tuple(conf.values()))
    env = _ProcgenEnvPool(env_spec)
    state_keys = env._state_keys
    env._reset(np.arange(num_envs, dtype=np.int32))
    total = 5000
    actions = np.random.randint(
      15, size=(total, batch)
    )  # procgen action lies in range 0~14 inclusively
    t = time.time()
    same_state_count = 0
    reset_count = 0
    total_reward = 0.0
    prev_state = None
    for i in range(total):
      state = dict(zip(state_keys, env._recv()))
      action = {
        "env_id": state["info:env_id"],
        "players.env_id": state["info:players.env_id"],
        "action": actions[i],
      }
      # TODO: `cv2.imwrite` dumpy the rgb pixels to picture for sanity check
      # if (i < 50):
      #   rgb_to_picture(state)
      if (
        prev_state is not None and
        np.allclose(state["obs:obs"], prev_state["obs:obs"])
      ):
        same_state_count += 1
      reset_count += int(state["done"][0] == True)
      total_reward += float(state["reward"][0])
      env._send(tuple(action.values()))
      prev_state = state
    duration = time.time() - t
    fps = total * batch / duration
    logging.info(
      f"Total steps {total} yieds {same_state_count} times of same prev/curr states, {reset_count} times of reset and total rewards of {total_reward}"
    )
    logging.info(f"Raw envpool Procgen FPS = {fps:.6f}")

  def test_gym_dm_align(self) -> None:
    # Make sure gym's envpool and dm_env's envpool generate the same data
    total = 1000
    num_envs = 4
    config = ProcgenEnvSpec.gen_config(num_envs=num_envs)
    spec = ProcgenEnvSpec(config)
    env0 = ProcgenGymEnvPool(spec)
    env1 = ProcgenDMEnvPool(spec)
    obs0 = env0.reset()
    obs1 = env1.reset().observation.obs
    np.testing.assert_allclose(
      np.array(obs0["obs"], dtype=float), np.array(obs1, dtype=float)
    )
    for _ in range(total):
      action = np.random.randint(15, size=num_envs)
      obs0 = env0.step(action)[0]
      obs1 = env1.step(action).observation.obs
      np.testing.assert_allclose(
        np.array(obs0["obs"], dtype=float), np.array(obs1, dtype=float)
      )

  def gym_check(
    self,
    game_name: str,
    spec_cls: Any,
    envpool_cls: Any,
    num_envs: int = 4
  ) -> None:
    env0 = envpool_cls(
      spec_cls(
        spec_cls.gen_config(num_envs=num_envs, seed=0, game_name=game_name)
      )
    )
    env1 = envpool_cls(
      spec_cls(
        spec_cls.gen_config(num_envs=num_envs, seed=0, game_name=game_name)
      )
    )
    env2 = envpool_cls(
      spec_cls(
        spec_cls.gen_config(num_envs=num_envs, seed=1, game_name=game_name)
      )
    )
    act_space = env0.action_space
    eps = np.finfo(np.float32).eps
    obs_space = env0.observation_space
    obs_min = 0.0 - eps
    obs_max = 255.0 + eps
    total = 200
    close, not_close = 0, 0
    for _ in range(total):
      action = np.array([act_space.sample() for _ in range(num_envs)])
      obs0 = env0.step(action)[0]["obs"][0]
      obs1 = env1.step(action)[0]["obs"][0]
      obs2 = env2.step(action)[0]["obs"][0]
      if (np.allclose(obs0, obs1)):
        close += 1
      if (not np.allclose(obs0, obs2)):
        not_close += 1
      np.testing.assert_allclose(obs0, obs1)
      self.assertFalse(np.allclose(obs0, obs2))
      self.assertTrue(np.all(obs_min <= obs0), obs0)
      self.assertTrue(np.all(obs_min <= obs2), obs2)
      self.assertTrue(np.all(obs0 <= obs_max), obs0)
      self.assertTrue(np.all(obs2 <= obs_max), obs2)
    logging.info(
      f"Gym {game_name}: among {total} steps taken, obs0 and obs1 equal for {close} times, "
      f"obs0 and obs2 differ by {not_close} times"
    )

  def dmc_check(
    self,
    game_name,
    spec_cls: Any,
    envpool_cls: Any,
    num_envs: int = 4,
  ) -> None:
    np.random.seed(0)
    env0 = envpool_cls(
      spec_cls(
        spec_cls.gen_config(num_envs=num_envs, seed=0, game_name=game_name)
      )
    )
    env1 = envpool_cls(
      spec_cls(
        spec_cls.gen_config(num_envs=num_envs, seed=0, game_name=game_name)
      )
    )
    env2 = envpool_cls(
      spec_cls(
        spec_cls.gen_config(num_envs=num_envs, seed=1, game_name=game_name)
      )
    )
    act_spec = env0.action_spec()
    total = 200
    close, not_close = 0, 0
    for t in range(total):
      action = np.array(
        [
          np.random.uniform(
            low=ACTION_LOW, high=ACTION_HIGH, size=act_spec.shape
          ) for _ in range(num_envs)
        ]
      )
      obs0 = env0.step(action).observation.obs[0]
      obs1 = env1.step(action).observation.obs[0]
      obs2 = env2.step(action).observation.obs[0]
      if (np.allclose(obs0, obs1)):
        close += 1
      if (not np.allclose(obs0, obs2)):
        not_close += 1
      np.testing.assert_allclose(obs0, obs1)
      self.assertFalse(np.allclose(obs0, obs2))
    logging.info(
      f"DMC {game_name}: among {total} steps taken, obs0 and obs1 equal for {close} times, "
      f"obs0 and obs2 differ by {not_close} times"
    )

  def test_gym_deterministic(self) -> None:
    # iterate over all procgen games to test Gym deterministic
    for game in procgen_games_list:
      self.gym_check(game, ProcgenEnvSpec, ProcgenGymEnvPool)

  def test_dmc_deterministic(self) -> None:
    # iterate over all procgen games to test DMC deterministic
    for game in procgen_games_list:
      self.dmc_check(game, ProcgenEnvSpec, ProcgenDMEnvPool)


if __name__ == "__main__":
  absltest.main()
