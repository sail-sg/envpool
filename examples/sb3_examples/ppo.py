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

from typing import Optional

import gym
import numpy as np
import torch as th
from packaging import version
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import (
  VecEnvObs,
  VecEnvStepReturn,
)

import envpool
from envpool.python.protocol import EnvPool

# Force PyTorch to use only one threads
# make things faster for simple envs
th.set_num_threads(1)

num_envs = 4
env_id = "Pendulum-v1"  # "CartPole-v1"
seed = 0
use_env_pool = True  # whether to use EnvPool or Gym for training
render = False  # whether to render final policy using Gym
is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")


class VecAdapter(VecEnvWrapper):
  """
  Convert EnvPool object to a Stable-Baselines3 (SB3) VecEnv.

  :param venv: The envpool object.
  """

  def __init__(self, venv: EnvPool):
    # Retrieve the number of environments from the config
    venv.num_envs = venv.spec.config.num_envs
    super().__init__(venv=venv)

  def step_async(self, actions: np.ndarray) -> None:
    self.actions = actions

  def reset(self) -> VecEnvObs:
    if is_legacy_gym:
      return self.venv.reset()
    else:
      return self.venv.reset()[0]

  def seed(self, seed: Optional[int] = None) -> None:
    # You can only seed EnvPool env by calling envpool.make()
    pass

  def step_wait(self) -> VecEnvStepReturn:
    if is_legacy_gym:
      obs, rewards, dones, info_dict = self.venv.step(self.actions)
    else:
      obs, rewards, terms, truncs, info_dict = self.venv.step(self.actions)
      dones = terms + truncs
    infos = []
    # Convert dict to list of dict
    # and add terminal observation
    for i in range(self.num_envs):
      infos.append(
        {
          key: info_dict[key][i]
          for key in info_dict.keys()
          if isinstance(info_dict[key], np.ndarray)
        }
      )
      if dones[i]:
        infos[i]["terminal_observation"] = obs[i]
        if is_legacy_gym:
          obs[i] = self.venv.reset(np.array([i]))
        else:
          obs[i] = self.venv.reset(np.array([i]))[0]
    return obs, rewards, dones, infos


if use_env_pool:
  env = envpool.make(env_id, env_type="gym", num_envs=num_envs, seed=seed)
  env.spec.id = env_id
  env = VecAdapter(env)
  env = VecMonitor(env)
else:
  env = make_vec_env(env_id, n_envs=num_envs)

# Tuned hyperparams for Pendulum-v1, works also for CartPole-v1
kwargs = {}
if env_id == "Pendulum-v1":
  # Use gSDE for better results
  kwargs = dict(use_sde=True, sde_sample_freq=4)

model = PPO(
  "MlpPolicy",
  env,
  n_steps=1024,
  learning_rate=1e-3,
  gae_lambda=0.95,
  gamma=0.9,
  verbose=1,
  seed=seed,
  **kwargs
)

# You can stop the training early by pressing ctrl + c
try:
  model.learn(100_000)
except KeyboardInterrupt:
  pass

# Agent trained on envpool version should also perform well on regular Gym env
test_env = gym.make(env_id)

def legacy_wrap(env):
  env.reset_fn = env.reset
  env.step_fn = env.step
  def legacy_reset():
    return env.reset_fn()[0]
  def legacy_step(action):
    obs, rew, term, trunc, info = env.step_fn(action)
    return obs, rew, term + trunc, info
  env.reset = legacy_reset
  env.step = legacy_step
  return env
if not is_legacy_gym:
  test_env = legacy_wrap(test_env)

# Test with EnvPool
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"EnvPool - {env_id}")
print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Test with Gym
mean_reward, std_reward = evaluate_policy(
  model,
  test_env,
  n_eval_episodes=20,
  warn=False,
  render=render,
)
print(f"Gym - {env_id}")
print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
