# Copyright 2022 Garena Online Private Limited
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
"""Helpers functions and wrappers."""

import logging
from typing import Iterable, Iterator, List, Mapping, Optional, Union

import dm_env
import gym
import jax
import jax.numpy as jnp
import numpy as np
import reverb
import tensorflow as tf
import tree
from acme import core, specs, types, wrappers
from acme.adders import Adder
from acme.adders.reverb import base as reverb_base
from acme.adders.reverb import sequence as reverb_sequence
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors, ppo
from acme.agents.jax.actor_core import (
  ActorCore,
  FeedForwardPolicyWithExtra,
  SimpleActorCoreStateWithExtras,
)
from acme.jax import networks as networks_lib
from acme.jax import utils, variable_utils
from acme.jax.types import PRNGKey
from acme.utils import loggers
from acme.utils.loggers import aggregators, base, filters, terminal
from packaging import version
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

import envpool
from envpool.python.protocol import EnvPool

logging.getLogger().setLevel(logging.INFO)
is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")


class TimeStep(dm_env.TimeStep):

  def last(self) -> bool:
    """Adapt the batch step_type for EnvironmentLoop episode-end checking."""
    return any(self.step_type == dm_env.StepType.LAST)


class BatchSequenceAdder(reverb_sequence.SequenceAdder):

  @classmethod
  def signature(
    cls,
    environment_spec: specs.EnvironmentSpec,
    extras_spec: types.NestedSpec = ...,
    sequence_length: Optional[int] = None,
    batch_size: Optional[int] = None,
  ):
    # Add env batch and time dimension.
    def add_extra_dim(paths: Iterable[str], spec: tf.TensorSpec):
      name = "/".join(str(p) for p in paths)
      if "rewards" in name:
        shape = (sequence_length, *spec.shape)
      else:
        shape = (sequence_length, batch_size, *spec.shape)
      return tf.TensorSpec(shape=shape, dtype=spec.dtype, name=name)

    trajectory_env_spec, trajectory_extras_spec = tree.map_structure_with_path(
      add_extra_dim, (environment_spec, extras_spec)
    )

    spec_step = reverb_base.Trajectory(
      *trajectory_env_spec,
      start_of_episode=tf.TensorSpec(
        shape=(sequence_length, batch_size),
        dtype=tf.bool,
        name="start_of_episode"
      ),
      extras=trajectory_extras_spec,
    )

    return spec_step


class AdderWrapper(Adder):

  def __init__(self, adder: Adder) -> None:
    self._adder: Adder = adder

  def reset(self):
    self._adder.reset()

  def add_first(self, timestep: TimeStep):
    if not any(timestep.first()):
      raise ValueError(
        "adder.add_first with an initial timestep (i.e. one for "
        "which timestep.first() is True"
      )
    # Record the next observation but leave the history buffer row open by
    # passing `partial_step=True`.
    self._adder._writer.append(
      dict(
        observation=timestep.observation, start_of_episode=timestep.first()
      ),
      partial_step=True,
    )
    self._adder._add_first_called = True

  def add(
    self,
    action: types.NestedArray,
    next_timestep: TimeStep,
    extras: types.NestedArray = ...,
  ):
    next_timestep = TimeStep(
      step_type=next_timestep.step_type,
      observation=next_timestep.observation,
      reward=next_timestep.reward,
      discount=next_timestep.discount,
    )
    self._adder.add(action, next_timestep, extras)


def _batched_feed_forward_with_extras_to_actor_core(
  policy: FeedForwardPolicyWithExtra
) -> ActorCore[SimpleActorCoreStateWithExtras, Mapping[str, jnp.ndarray]]:
  """Modified adapter allowing batched data processing."""

  def select_action(
    params: networks_lib.Params,
    observation: networks_lib.Observation,
    state: SimpleActorCoreStateWithExtras,
  ):
    rng = state.rng
    rng1, rng2 = jax.random.split(rng)
    action, extras = policy(params, rng1, observation)
    return action, SimpleActorCoreStateWithExtras(rng2, extras)

  def init(rng: PRNGKey) -> SimpleActorCoreStateWithExtras:
    return SimpleActorCoreStateWithExtras(rng, {})

  def get_extras(
    state: SimpleActorCoreStateWithExtras
  ) -> Mapping[str, jnp.ndarray]:
    return state.extras

  return ActorCore(
    init=init, select_action=select_action, get_extras=get_extras
  )


class PPOBuilder(ppo.PPOBuilder):
  """Wrap the PPO algorithm builder for EnvPool."""

  def __init__(self, config: ppo.PPOConfig, num_envs: int = -1):
    super().__init__(config)
    self._num_envs = num_envs
    self._batch_env = num_envs > 0

  def make_replay_tables(
    self, environment_spec: specs.EnvironmentSpec,
    policy: actor_core_lib.FeedForwardPolicyWithExtra
  ) -> List[reverb.Table]:
    if not self._batch_env:
      return super(PPOBuilder,
                   self).make_replay_tables(environment_spec, policy)
    extra_spec = {
      "log_prob": np.ones(shape=(), dtype=np.float32),
    }
    signature = BatchSequenceAdder.signature(
      environment_spec,
      extra_spec,
      sequence_length=self._sequence_length,
      batch_size=self._num_envs,
    )
    return [
      reverb.Table.queue(
        name=self._config.replay_table_name,
        max_size=self._config.batch_size // self._num_envs,
        signature=signature,
      )
    ]

  def make_dataset_iterator(
    self, replay_client: reverb.Client
  ) -> Iterator[reverb.ReplaySample]:
    if not self._batch_env:
      return super(PPOBuilder, self).make_dataset_iterator(replay_client)
    assert self._config.batch_size % self._num_envs == 0
    batch_size = self._config.batch_size // self._num_envs
    dataset = reverb.TrajectoryDataset.from_table_signature(
      server_address=replay_client.server_address,
      table=self._config.replay_table_name,
      max_in_flight_samples_per_worker=2 * batch_size,
    )

    def transpose(sample: reverb.ReplaySample) -> reverb.ReplaySample:
      _data = sample.data

      def _process(data):
        shape = data.shape
        data = tf.transpose(
          data, (0, 2, 1, *[i for i in range(3, len(shape))])
        )
        data = tf.reshape(
          data, (
            self._config.batch_size, self._config.unroll_length + 1, *shape[3:]
          )
        )
        return data

      return reverb.ReplaySample(
        info=sample.info,
        data=tree.map_structure(_process, _data),
      )

    # Add batch dimension.
    dataset = dataset.batch(batch_size, drop_remainder=True).map(transpose)
    return utils.device_put(dataset.as_numpy_iterator(), jax.devices()[0])

  def make_adder(self, replay_client: reverb.Client) -> Adder:
    adder = super(PPOBuilder, self).make_adder(replay_client)
    if not self._batch_env:
      return adder
    return AdderWrapper(adder)

  def make_actor(
    self,
    random_key: networks_lib.PRNGKey,
    policy_network: actor_core_lib.FeedForwardPolicyWithExtra,
    environment_spec: specs.EnvironmentSpec,
    variable_source: Optional[core.VariableSource] = None,
    adder: Optional[Adder] = None,
  ) -> core.Actor:
    if not self._batch_env:
      return super().make_actor(
        random_key, policy_network, environment_spec, variable_source, adder
      )
    assert variable_source is not None
    variable_client = variable_utils.VariableClient(
      variable_source,
      "network",
      device="cpu",
      update_period=self._config.variable_update_period,
    )
    actor = _batched_feed_forward_with_extras_to_actor_core(policy_network)
    return actors.GenericActor(
      actor, random_key, variable_client, adder, backend="cpu"
    )


class BatchEnvWrapper(dm_env.Environment):

  def __init__(
    self,
    environment: Union[DummyVecEnv, EnvPool],
  ):
    self._environment = environment
    if not isinstance(environment, DummyVecEnv):
      self._num_envs = len(environment.all_env_ids)
      self._use_env_pool = True
    else:
      self._num_envs = environment.num_envs
      self._use_env_pool = False
    self._reset_next_step = True

  def reset(self) -> TimeStep:
    self._reset_next_step = False
    if is_legacy_gym:
      observation = self._environment.reset()
    else:
      observation, _ = self._environment.reset()
    ts = TimeStep(
      step_type=np.full(self._num_envs, dm_env.StepType.FIRST, dtype="int32"),
      reward=np.zeros(self._num_envs, dtype="float32"),
      discount=np.ones(self._num_envs, dtype="float32"),
      observation=observation,
    )
    return ts

  def step(self, action: types.NestedArray) -> TimeStep:
    if self._reset_next_step:
      return self.reset()
    if self._use_env_pool:
      if is_legacy_gym:
        observation, reward, done, _ = self._environment.step(action)
      else:
        observation, reward, term, trunc, _ = self._environment.step(action)
        done = term + trunc
    else:
      self._environment.step_async(action)
      observation, reward, done, _ = self._environment.step_wait()
    self._reset_next_step = any(done)
    ts = TimeStep(
      step_type=(done + 1).astype(np.int32),
      reward=reward,
      discount=(1 - done).astype(np.float32),
      observation=observation,
    )
    return ts

  def observation_spec(self):
    space = self._environment.observation_space
    obs_spec = specs.BoundedArray(
      shape=space.shape,
      dtype="float32",
      minimum=space.low,
      maximum=space.high,
      name="observation",
    )
    return obs_spec

  def action_spec(self):
    space = self._environment.action_space
    if isinstance(space, gym.spaces.Discrete):
      act_spec = specs.DiscreteArray(
        num_values=space.n, dtype=space.dtype, name="action"
      )
      return act_spec
    return specs.BoundedArray(
      shape=space.shape,
      dtype=space.dtype,
      minimum=space.low,
      maximum=space.high,
      name="single_action",
    )

  def reward_spec(self):
    return specs.Array(
      name="reward",
      shape=[self._num_envs],
      dtype="float32",
    )

  def close(self):
    self._environment.close()


def make_mujoco_environment(
  task: str, use_envpool: bool = False, use_vec_env=False, num_envs: int = 2
):
  env_wrappers = []
  if use_envpool:
    env = envpool.make(
      task,
      env_type="gym",
      num_envs=num_envs,
    )
    env_wrappers.append(BatchEnvWrapper)
  elif use_vec_env:
    env = make_vec_env(task, n_envs=num_envs)
    env_wrappers.append(BatchEnvWrapper)
  else:
    env = gym.make(task)
    env_wrappers.append(wrappers.GymWrapper)
  env_wrappers.append(wrappers.SinglePrecisionWrapper)
  return wrappers.wrap_all(env, env_wrappers)


def make_logger(
  label: str,
  config: dict,
  run_name: str = "",
  wb_entity: str = "",
  steps_key: str = "steps",
  task_instance: int = 0,
) -> loggers.Logger:
  del task_instance, steps_key
  num_envs = config["num_envs"] if config["use_batch_env"] else 1

  print_fn = logging.info
  terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)
  logger = [terminal_logger]

  if label == "train":

    import wandb

    class WBLogger(loggers.Logger):

      def __init__(self, num_envs) -> None:
        self._num_envs = num_envs

      def write(self, data) -> None:
        new_data = {}
        for key, value in data.items():
          if key in ["train_steps", "actor_steps"]:
            key = "global_step"
            value *= self._num_envs
          elif config["use_batch_env"] and key == "episode_return":
            value = value[0]
          new_data[key] = value
        wandb.log(new_data)

      def close(self) -> None:
        wandb.finish()

    wandb.init(
      project=config["wb_project"],
      entity=wb_entity,
      name=run_name,
      config=config,
    )
    logger.append(WBLogger(num_envs))

  # Dispatch to all writers and filter Nones and by time.
  logger = aggregators.Dispatcher(logger, base.to_numpy)
  logger = filters.NoneFilter(logger)

  return logger
