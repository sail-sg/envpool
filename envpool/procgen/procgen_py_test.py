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
from typing import Any

import cv2
import dm_env
import gym
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

# in total 16 games name in Procgen
procgen_games_list = [
  "bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun",
  "dodgeball", "fruitbot", "heist", "jumper", "leaper", "maze", "miner",
  "ninja", "plunder", "starpilot"
]

# 16 games has different timeout settings on their own
procgen_timeout_list = {
  "bigfish": 6000,
  "bossfight": 4000,
  "caveflyer": 1000,
  "chaser": 1000,
  "climber": 1000,
  "coinrun": 1000,
  "dodgeball": 1000,
  "fruitbot": 1000,
  "heist": 1000,
  "jumper": 1000,
  "leaper": 500,
  "maze": 500,
  "miner": 1000,
  "ninja": 1000,
  "plunder": 4000,
  "starpilot": 1000
}


def rgb_to_picture(pixels, count=pic_count, prefix_name="procgen"):
  # convert a state's rgb 64x64x3 game observation into picture by cv2
  # for sanity check if the game is running correctly
  # state is ordered in y -> x -> rgb in one dimension array
  global pic_count
  os.makedirs("img", exist_ok=True)
  index = 0
  img = np.zeros((RES_W, RES_H, RGB_FACTOR), np.uint8)
  for h in range(RES_W):
    for w in range(RES_H):
      red = pixels[index]
      green = pixels[index + 1]
      blue = pixels[index + 2]
      index += 3
      # cv2's ordering is BGR
      img[h][w][0] = blue
      img[h][w][1] = green
      img[h][w][2] = red
  pic_path = f"img/{prefix_name}_{count:03d}.png"
  print("Outwrite", os.getcwd() + "/" + pic_path)
  cv2.imwrite(pic_path, img)


class _ProcgenEnvPoolTest(absltest.TestCase):

  def test_config(self) -> None:
    # test the config key is same as what we expect
    ref_config_keys = [
      "action_num", "base_path", "batch_size", "distribution_mode",
      "game_name", "use_sequential_levels", "max_num_players", "num_envs",
      "num_levels", "num_threads", "seed", "start_level", "state_num",
      "gym_reset_return_info", "thread_affinity_offset"
    ]
    default_conf = _ProcgenEnvSpec._default_config_values
    self.assertTrue(isinstance(default_conf, tuple))
    config_keys = _ProcgenEnvSpec._config_keys
    self.assertTrue(isinstance(config_keys, list))
    self.assertEqual(len(default_conf), len(config_keys))
    self.assertEqual(sorted(config_keys), sorted(ref_config_keys))

  def test_raw_envpool(self) -> None:
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
    total = 1000
    actions = np.random.randint(15, size=(total, batch))
    t = time.time()
    for i in range(total):
      state = dict(zip(state_keys, env._recv()))
      action = {
        "env_id": state["info:env_id"],
        "players.env_id": state["info:players.env_id"],
        "action": actions[i],
      }
      # if (i < 100):
      # #   output the first few steps to picture for animation
      # #   to check if the game is moving as we expect
      #   rgb_to_picture(state["obs:obs"][0], i, "procgen)
      env._send(tuple(action.values()))
    duration = time.time() - t
    fps = total * batch / duration
    logging.info(f"Raw envpool Procgen FPS = {fps:.6f}")

  def gym_deterministic_check(
    self,
    game_name: str,
    spec_cls: Any,
    envpool_cls: Any,
    num_envs: int = 4
  ) -> None:
    logging.info(f"deterministic check for gym {game_name}")
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

  def gym_align_check(self, game_name, spec_cls: Any, envpool_cls: Any):
    logging.info(f"align check for gym {game_name}")
    timeout = procgen_timeout_list[game_name]
    num_env = 1
    for i in range(5):
      env_gym = envpool_cls(
        spec_cls(
          spec_cls.gen_config(num_envs=num_env, seed=i, game_name=game_name)
        )
      )
      env_procgen = gym.make(
        f"procgen:procgen-{game_name}-v0",
        rand_seed=i,
        use_generated_assets=True
      )
      env_gym.reset(np.arange(num_env, dtype=np.int32))
      env_procgen.reset()
      act_space = env_procgen.action_space
      envpool_done = False
      cnt = 1
      while (not envpool_done and cnt < timeout):
        cnt += 1
        action = np.array([act_space.sample() for _ in range(num_env)])
        _, raw_reward, raw_done, _ = env_procgen.step(action[0])
        _, envpool_reward, envpool_done, _, = env_gym.step(action)
        envpool_reward, envpool_done = envpool_reward[0], envpool_done[0]
        # must die and earn reward same time aligned
        self.assertTrue(envpool_reward == raw_reward)
        self.assertTrue(raw_done == envpool_done)

  def dmc_deterministic_check(
    self,
    game_name,
    spec_cls: Any,
    envpool_cls: Any,
    num_envs: int = 4,
  ) -> None:
    logging.info(f"deterministic check for dmc {game_name}")
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
    for _ in range(total):
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

  def dmc_align_check(self, game_name, spec_cls: Any, envpool_cls: Any):
    logging.info(f"align check for dmc {game_name}")
    timeout = procgen_timeout_list[game_name]
    num_env = 1
    for i in range(5):
      env_dmc = envpool_cls(
        spec_cls(
          spec_cls.gen_config(num_envs=num_env, seed=i, game_name=game_name)
        )
      )
      env_procgen = gym.make(
        f"procgen:procgen-{game_name}-v0",
        rand_seed=i,
        use_generated_assets=True
      )
      env_procgen.reset()
      act_space = env_procgen.action_space
      envpool_done = False
      cnt = 1
      while (not envpool_done and cnt < timeout):
        cnt += 1
        action = np.array([act_space.sample() for _ in range(num_env)])
        _, raw_reward, raw_done, _ = env_procgen.step(action[0])
        r = env_dmc.step(action)
        envpool_reward, envpool_done = r.reward[
          0], r.step_type == dm_env.StepType.LAST
        # must die and earn reward same time aligned
        self.assertTrue(envpool_reward == raw_reward)
        self.assertTrue(raw_done == envpool_done)

  def test_gym_deterministic(self) -> None:
    # iterate over all procgen games to test Gym deterministic
    for game in procgen_games_list:
      self.gym_deterministic_check(game, ProcgenEnvSpec, ProcgenGymEnvPool)

  def test_gym_align(self) -> None:
    # iterate over all procgen games to test Gym align
    for game in procgen_games_list:
      self.gym_align_check(game, ProcgenEnvSpec, ProcgenGymEnvPool)

  def test_dmc_deterministic(self) -> None:
    # iterate over all procgen games to test DMC deterministic
    for game in procgen_games_list:
      self.dmc_deterministic_check(game, ProcgenEnvSpec, ProcgenDMEnvPool)

  def test_dmc_align(self) -> None:
    # iterate over all procgen games to test DMC align
    for game in procgen_games_list:
      self.dmc_align_check(game, ProcgenEnvSpec, ProcgenDMEnvPool)


if __name__ == "__main__":
  absltest.main()
