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
from typing import List

import acme
import dm_env
import numpy as np
import tree
from acme import specs, types
from acme.agents.jax import r2d2
from acme.utils import loggers
from acme.wrappers import EnvironmentWrapper
from acme.wrappers.observation_action_reward import OAR

import envpool
from envpool.python.protocol import EnvPool


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", type=str, default="Pong-v5")
  parser.add_argument('--epoch', type=int, default=100)
  parser.add_argument("--training-num", type=int, default=20)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--trace-length", type=int, default=20)
  parser.add_argument("--burn-in-length", type=int, default=10)
  parser.add_argument("--sequence-period", type=int, default=10)
  return parser.parse_args()


class EnvWrapper(EnvironmentWrapper):

  def __init__(self, environment: EnvPool):
    super().__init__(environment)

  def observation_spec(self) -> OAR:
    obs = self._environment.observation_spec().obs
    new_obs = dm_env.specs.BoundedArray(
      name=obs.name,
      shape=[s for s in obs.shape if s != -1],
      dtype="float32",
      minimum=obs.minimum,
      maximum=obs.maximum,
    )
    return OAR(
      observation=new_obs,
      action=self.action_spec(),
      reward=self.reward_spec()
    )

  def reset(self) -> List[OAR]:
    timestep = self._environment.reset()
    timesteps = self._split_timestep(timestep)

    # Initialize with zeros of the appropriate shape/dtype.
    action = tree.map_structure(
      lambda x: x.generate_value(), self._environment.action_spec()
    )
    reward = tree.map_structure(
      lambda x: x.generate_value(), self._environment.reward_spec()
    )

    new_timesteps = [
      self._augment_observation(action, reward, ts) for ts in timesteps
    ]
    return new_timesteps

  def step(self, action: np.ndarray, env_id: List[str]) -> List[OAR]:
    timestep = self._environment.step(action, env_id)
    timesteps = self._split_timestep(timestep)
    new_timesteps = [
      self._augment_observation(action[i], timesteps[i].reward, timesteps[i])
      for i in range(len(timesteps))
    ]
    return new_timesteps

  def _split_timestep(self,
                      timestep: dm_env.TimeStep) -> List[dm_env.TimeStep]:
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

  def _augment_observation(
    self, action: types.NestedArray, reward: types.NestedArray,
    timestep: dm_env.TimeStep
  ) -> dm_env.TimeStep:
    oar = OAR(observation=timestep.observation, action=action, reward=reward)
    return timestep._replace(observation=oar)


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

  config = r2d2.R2D2Config(
    batch_size=16, trace_length=20, burn_in_length=10, sequence_period=10
  )

  agent = r2d2.R2D2(
    env_spec,
    networks=r2d2.make_atari_networks(config.batch_size, env_spec),
    config=config,
    seed=args.seed
  )

  loop = EnvironmentLoopWrapper(train_envs, agent)
  loop.run(args.epoch)


if __name__ == '__main__':
  run_dqn(get_args())
