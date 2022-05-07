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

import argparse
import operator
import time

import acme
import dm_env
import numpy as np
import tree
from acme import specs
from acme.agents.tf import dqn
from acme.tf import networks
from acme.utils import loggers
from acme.wrappers import EnvironmentWrapper

import envpool


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", type=str, default="Pong-v5")
  parser.add_argument('--epoch', type=int, default=100)
  parser.add_argument("--training-num", type=int, default=20)
  return parser.parse_args()


class EnvWrapper(EnvironmentWrapper):

  def __init__(self, environment):
    super().__init__(environment)

  def observation_spec(self):
    obs = self._environment.observation_spec().obs
    new_obs = dm_env.specs.BoundedArray(
      name=obs.name,
      shape=[s for s in obs.shape if s != -1],
      dtype="float32",
      minimum=obs.minimum,
      maximum=obs.maximum,
    )
    return new_obs

  def reset(self):
    timestep = self._environment.reset()
    return self.split_timestep(timestep)

  def step(self, action, env_id):
    timestep = self._environment.step(action, env_id)
    return self.split_timestep(timestep)

  def split_timestep(self, timestep):
    timesteps = []
    for i in range(len(self._environment)):
      timesteps.append(
        dm_env.TimeStep(
          step_type=timestep.step_type[i],
          observation=timestep.observation.obs[i].astype("float32"),
          reward=np.array(timestep.reward[i]).astype("float64"),
          discount=np.array(timestep.discount[i]).astype("float64"),
        )
      )
    return timesteps


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)


class EnvironmentLoopWrapper(acme.EnvironmentLoop):

  def __init__(self, environment: dm_env.Environment, actor: acme.core.Actor):
    super().__init__(environment, actor)

  def run_episode(self) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to
    retrieve an action.

    Returns:
      An instance of `loggers.LoggingData`.
    """
    # Reset any counts and start the environment.
    start_time = time.time()
    episode_steps = 0

    # For evaluation, this keeps track of the total undiscounted reward
    # accumulated during the episode.
    episode_return = tree.map_structure(
      _generate_zeros_from_spec, self._environment.reward_spec()
    )
    timesteps = self._environment.reset()
    # Make the first observation.
    self._actor.observe_first(timesteps[0])
    for observer in self._observers:
      # Initialize the observer with the current state of the env after reset
      # and the initial timestep.
      for timestep in timesteps:
        observer.observe_first(self._environment, timestep)

    # Run an episode.
    while not any([timestep.last() for timestep in timesteps]):
      actions = []
      for timestep in timesteps:
        # Generate an action from the agent's policy and step the environment.
        actions.append(self._actor.select_action(timestep.observation))
      actions = np.array(actions)

      timesteps = self._environment.step(
        actions, self._environment.all_env_ids
      )

      # Have the agent observe the timestep and let the actor update itself.
      for action, timestep in zip(actions, timesteps):
        self._actor.observe(action, next_timestep=timestep)
        for observer in self._observers:
          # One environment step was completed. Observe the current state of
          # the environment, the current timestep and the action.
          observer.observe(self._environment, timestep, action)
        if timestep.last():
          break
      if self._should_update:
        self._actor.update()

      # Book-keeping.
      episode_steps += 1

      # Equivalent to: episode_return += timestep.reward
      # We capture the return value because if timestep.reward is a JAX
      # DeviceArray, episode_return will not be mutated in-place. (In all other
      # cases, the returned episode_return will be the same object as the
      # argument episode_return.)
      episode_return = tree.map_structure(
        operator.iadd, episode_return, timesteps[0].reward
      )

    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_steps)

    # Collect the results and combine with counts.
    steps_per_second = episode_steps / (time.time() - start_time)
    result = {
      'episode_length': episode_steps,
      'episode_return': episode_return,
      'steps_per_second': steps_per_second,
    }
    result.update(counts)
    for observer in self._observers:
      result.update(observer.get_metrics())
    return result


def run_dqn(args):
  train_envs = EnvWrapper(
    envpool.make(args.task, num_envs=args.training_num, env_type="dm")
  )
  env_spec = acme.make_environment_spec(train_envs)
  network = networks.DQNAtariNetwork(env_spec.actions.num_values)

  agent = dqn.DQN(env_spec, network)
  loop = EnvironmentLoopWrapper(train_envs, agent)
  loop.run(args.epoch)


if __name__ == '__main__':
  run_dqn(get_args())
