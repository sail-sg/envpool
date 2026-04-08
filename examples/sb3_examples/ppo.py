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

import warnings
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
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


class VecAdapter(VecEnv):
    """Convert EnvPool object to a Stable-Baselines3 (SB3) VecEnv.

    :param venv: The envpool object.
    """

    def __init__(self, venv: EnvPool):
        """Initialize the adapter around an EnvPool vector env."""
        self.venv = venv
        super().__init__(
            num_envs=venv.spec.config.num_envs,
            observation_space=venv.observation_space,
            action_space=venv.action_space,
        )

    def _info_dict_to_list(
        self, info_dict: dict[str, Any], batch_size: int
    ) -> list[dict[str, Any]]:
        """Convert EnvPool's batched info dict to SB3's list-of-dicts format."""
        array_info = {
            key: value
            for key, value in info_dict.items()
            if isinstance(value, np.ndarray)
        }
        infos = []
        for i in range(batch_size):
            infos.append({key: value[i] for key, value in array_info.items()})
        return infos

    def step_async(self, actions: np.ndarray) -> None:
        """Store the actions for the next environment step."""
        self.actions = actions

    def reset(self) -> VecEnvObs:
        """Reset the wrapped vector environment."""
        obs, info_dict = self.venv.reset()
        self.reset_infos = self._info_dict_to_list(info_dict, self.num_envs)
        self._reset_seeds()
        self._reset_options()
        return obs

    def seed(self, seed: int | None = None) -> list[None]:
        """Document that seeding happens when the environment is created."""
        return [None for _ in range(self.num_envs)]

    def step_wait(self) -> VecEnvStepReturn:
        """Step the wrapped environment and adapt the returned info."""
        obs, rewards, terms, truncs, info_dict = self.venv.step(self.actions)
        dones = terms | truncs
        infos = self._info_dict_to_list(info_dict, self.num_envs)
        for i in range(self.num_envs):
            if dones[i]:
                infos[i]["terminal_observation"] = obs[i]
                reset_obs, reset_info = self.venv.reset(np.array([i]))
                obs[i] = reset_obs[0]
                self.reset_infos[i] = self._info_dict_to_list(reset_info, 1)[0]
        return obs, rewards, dones, infos

    def close(self) -> None:
        """Close the wrapped EnvPool object."""
        self.venv.close()

    def get_attr(
        self, attr_name: str, indices: VecEnvIndices = None
    ) -> list[Any]:
        """Return attributes from EnvPool in SB3 VecEnv format."""
        attr = getattr(self.venv, attr_name)
        return [attr for _ in self._get_indices(indices)]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        """Set a shared EnvPool attribute for the selected vector indices."""
        for _ in self._get_indices(indices):
            setattr(self.venv, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        """Call an EnvPool method for each selected vector index."""
        method = getattr(self.venv, method_name)
        return [
            method(*method_args, **method_kwargs)
            for _ in self._get_indices(indices)
        ]

    def env_is_wrapped(
        self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> list[bool]:
        """EnvPool does not wrap individual Gymnasium env instances."""
        return [False for _ in self._get_indices(indices)]

    def get_images(self) -> list[np.ndarray | None]:
        """Return per-env rendered frames when EnvPool uses rgb_array mode."""
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but get_images() "
                "requires render_mode='rgb_array'.",
                stacklevel=2,
            )
            return [None for _ in range(self.num_envs)]
        frames = self.venv.render(env_ids=self.venv.all_env_ids)
        if frames is None:
            return [None for _ in range(self.num_envs)]
        return list(frames)


if use_env_pool:
    env = envpool.make(
        env_id, env_type="gymnasium", num_envs=num_envs, seed=seed
    )
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
