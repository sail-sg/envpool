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


"""Run Stable-Baselines3 PPO with EnvPool."""

import gymnasium as gym
import numpy as np
import torch as th
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
use_env_pool = True  # whether to use EnvPool or Gymnasium for training
render = False  # whether to render final policy using Gymnasium


class VecAdapter(VecEnvWrapper):
    """Convert EnvPool object to a Stable-Baselines3 (SB3) VecEnv.

    :param venv: The envpool object.
    """

    def __init__(self, venv: EnvPool):
        # Retrieve the number of environments from the config
        """Initialize the adapter around an EnvPool vector env."""
        venv.num_envs = venv.spec.config.num_envs
        super().__init__(venv=venv)

    def step_async(self, actions: np.ndarray) -> None:
        """Store the actions for the next environment step."""
        self.actions = actions

    def reset(self) -> VecEnvObs:
        """Reset the wrapped vector environment."""
        return self.venv.reset()[0]

    def seed(self, seed: int | None = None) -> None:
        # You can only seed EnvPool env by calling envpool.make()
        """Document that seeding happens when the environment is created."""
        pass

    def step_wait(self) -> VecEnvStepReturn:
        """Step the wrapped environment and adapt the returned info."""
        obs, rewards, terms, truncs, info_dict = self.venv.step(self.actions)
        dones = terms + truncs
        infos = []
        # Convert dict to list of dict
        # and add terminal observation
        for i in range(self.num_envs):
            infos.append({
                key: info_dict[key][i]
                for key in info_dict.keys()
                if isinstance(info_dict[key], np.ndarray)
            })
            if dones[i]:
                infos[i]["terminal_observation"] = obs[i]
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
    **kwargs,
)

# You can stop the training early by pressing ctrl + c
try:
    model.learn(100_000)
except KeyboardInterrupt:
    pass

# Agent trained on envpool version should also perform well on regular Gymnasium env.
test_env = gym.make(env_id)

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
print(f"Gymnasium - {env_id}")
print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
