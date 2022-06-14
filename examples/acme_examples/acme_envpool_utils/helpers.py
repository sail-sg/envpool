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
"""Helpers functions and wrappers."""
import copy
import dataclasses
import logging
import operator
import time
from functools import partial
from typing import (
  Any,
  Callable,
  Generic,
  Iterable,
  Iterator,
  List,
  Mapping,
  Optional,
  Sized,
  Tuple,
  Union,
)

import acme
import dm_env
import gym
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
import tensorflow as tf
import tree
from acme import core, specs, types, wrappers
from acme.adders import Adder
from acme.adders.reverb import base as reverb_base
from acme.adders.reverb import sequence as reverb_sequence
from acme.adders.reverb import utils as reverb_utils
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors, builders, impala, ppo
from acme.agents.jax.actor_core import (
  ActorCore,
  FeedForwardPolicyWithExtra,
  SimpleActorCoreStateWithExtras,
)
from acme.agents.jax.impala import types as impala_types

# from acme.agents.jax.impala.acting import IMPALAActor
from acme.datasets.reverb import make_reverb_dataset
from acme.jax import networks as networks_lib
from acme.jax import utils, variable_utils
from acme.jax.networks import atari
from acme.jax.types import PRNGKey
from acme.utils import counting, loggers
from acme.utils.loggers import aggregators, base, filters, terminal

# from acme.agents.jax.impala import types
from acme.wrappers import observation_action_reward
from acme.wrappers.observation_action_reward import OAR
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

import envpool
from envpool.python.protocol import EnvPool

logging.getLogger().setLevel(logging.INFO)


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
    batch_size: Optional[int] = None
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
      extras=trajectory_extras_spec
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
      partial_step=True
    )
    self._adder._add_first_called = True

  def add(
    self,
    action: types.NestedArray,
    next_timestep: TimeStep,
    extras: types.NestedArray = ...
  ):
    next_timestep = TimeStep(
      step_type=next_timestep.step_type,
      observation=next_timestep.observation,
      reward=next_timestep.reward,
      discount=next_timestep.discount,
    )
    self._adder.add(action, next_timestep, extras)


class AutoResetSequenceAdder(Adder):

  def __init__(
    self,
    client: reverb.Client,
    sequence_length: int,
    period: int,
    max_in_flight_items: Optional[int] = 2,
    priority_fns: Optional[reverb_base.PriorityFnMapping] = None,
    validate_items: bool = True,
  ) -> None:
    self._client = client
    self._sequence_length = sequence_length
    self._period = period
    self._max_sequence_length = sequence_length + 1
    self._max_in_flight_items = max_in_flight_items
    self._validate_items = validate_items
    self._timestep = None
    if priority_fns:
      priority_fns = dict(priority_fns)
    else:
      priority_fns = {reverb_base.DEFAULT_PRIORITY_TABLE: None}
    self._priority_fns = priority_fns
    self.__writer = None

  @property
  def _writer(self) -> reverb.TrajectoryWriter:
    if self.__writer is None:
      self.__writer = self._client.trajectory_writer(
        num_keep_alive_refs=self._max_sequence_length,
        validate_items=self._validate_items
      )
      self._writer_created_timestamp = time.time()
    return self.__writer

  def _maybe_create_item(self):
    first_write = self._writer.episode_steps == self._sequence_length
    period_reached = (
      self._writer.episode_steps > self._sequence_length and (
        (self._writer.episode_steps - self._sequence_length) % self._period
        == 0
      )
    )
    if not first_write and not period_reached:
      print(self._writer.episode_steps, "return!")
      return
    print(self._writer.episode_steps)
    # import pdb;pdb.set_trace()

    get_traj = operator.itemgetter(slice(-self._sequence_length, None))

    trajectory = reverb_base.Trajectory(
      **tree.map_structure(get_traj, self._writer.history)
    )
    table_priorities = reverb_utils.calculate_priorities(
      self._priority_fns, trajectory
    )
    for table_name, priority in table_priorities.items():
      self._writer.create_item(table_name, priority, trajectory)
      self._writer.flush(self._max_in_flight_items)

  def add_first(self, timestep: dm_env.TimeStep):
    self._timestep = timestep

  def add(
    self,
    action: types.NestedArray,
    next_timestep: dm_env.TimeStep,
    extras: types.NestedArray = ...
  ):
    assert self._timestep is not None
    has_extras = (
      len(extras) > 0 if isinstance(extras, Sized) \
        else extras is not None
    )
    transition = dict(
      observation=self._timestep.observation,
      action=action,
      reward=next_timestep.reward,
      discount=next_timestep.discount,
      start_of_episode=self._timestep.step_type == dm_env.StepType.FIRST,
      **({
        "extras": extras
      } if has_extras else {})
    )
    self._writer.append(transition)
    self._timestep = next_timestep
    self._maybe_create_item()

  def reset(self):
    return super().reset()

  def __del__(self):
    if self.__writer is not None:
      timeout_ms = 10_000
      # Try flush all appended data before closing to avoid loss of experience.
      try:
        self.__writer.flush(0, timeout_ms=timeout_ms)
      except reverb.DeadlineExceededError as e:
        logging.error(
          "Timeout (%d ms) exceeded when flushing the writer before "
          "deleting it. Caught Reverb exception: %s", timeout_ms, str(e)
        )
      self.__writer.close()


def _batched_feed_forward_with_extras_to_actor_core(
  policy: FeedForwardPolicyWithExtra
) -> ActorCore[SimpleActorCoreStateWithExtras, Mapping[str, jnp.ndarray]]:
  """Modified adapter allowing batched data processing."""

  def select_action(
    params: networks_lib.Params, observation: networks_lib.Observation,
    state: SimpleActorCoreStateWithExtras
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
      batch_size=self._num_envs
    )
    return [
      reverb.Table.queue(
        name=self._config.replay_table_name,
        max_size=self._config.batch_size // self._num_envs,
        signature=signature
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
      max_in_flight_samples_per_worker=2 * batch_size
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
        info=sample.info, data=tree.map_structure(_process, _data)
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
      update_period=self._config.variable_update_period
    )
    actor = _batched_feed_forward_with_extras_to_actor_core(policy_network)
    return actors.GenericActor(
      actor, random_key, variable_client, adder, backend="cpu"
    )


class IMPALAActor(acme.Actor):
  """A recurrent actor for batch inference."""

  _state: hk.LSTMState
  _prev_state: hk.LSTMState
  _prev_logits: jnp.ndarray

  def __init__(
    self,
    forward_fn: impala_types.PolicyValueFn,
    initial_state_fn: impala_types.RecurrentStateFn,
    rng: hk.PRNGSequence,
    batch_size: int,
    variable_client: Optional[variable_utils.VariableClient] = None,
    adder: Optional[Adder] = None,
  ):

    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._forward = forward_fn
    self._reset_fn_or_none = getattr(forward_fn, 'reset', None)
    self._rng = rng
    self._batch_size = batch_size

    self._initial_state = initial_state_fn(next(self._rng))

  def select_action(
    self, observation: impala_types.Observation
  ) -> impala_types.Action:

    if self._state is None:
      batch_state = [self._initial_state for _ in range(self._batch_size)]
      tree_def = tree.map_structure(lambda x: None, batch_state[0])
      batch_state = list(zip(*batch_state))
      batch_state = tree.unflatten_as(tree_def, batch_state)
      self._state = tree.map_structure_up_to(
        tree_def, lambda x: jnp.stack(x, axis=0), batch_state
      )

    # Forward.
    (logits,
     _), new_state = self._forward(self._params, observation, self._state)

    self._prev_logits = logits
    self._prev_state = self._state
    self._state = new_state

    action = jax.random.categorical(next(self._rng), logits)

    return action

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

    # Set the state to None so that we re-initialize at the next policy call.
    self._state = None

    # Reset state of inference functions that employ stateful wrappers (eg. BIT)
    # at the start of the episode.
    if self._reset_fn_or_none is not None:
      self._reset_fn_or_none()

  def observe(
    self,
    action: impala_types.Action,
    next_timestep: dm_env.TimeStep,
  ):
    # We re-initialize the recurrent state.
    re_init_idx = np.argwhere(next_timestep.step_type == dm_env.StepType.LAST
                             ).squeeze(-1)
    if re_init_idx.size > 0:
      for i in re_init_idx:
        self._state = tree.map_structure(lambda x: x.at[i].set(0), self._state)

    if not self._adder:
      return

    extras = {'logits': self._prev_logits, 'core_state': self._prev_state}
    self._adder.add(action, next_timestep, extras)

  def update(self, wait: bool = False):
    if self._variable_client is not None:
      self._variable_client.update(wait)

  @property
  def _params(self) -> Optional[hk.Params]:
    if self._variable_client is None:
      # If self._variable_client is None then we assume self._forward  does not
      # use the parameters it is passed and just return None.
      return None
    return self._variable_client.params


class IMPALABuilder(
  builders.ActorLearnerBuilder[impala.IMPALANetworks, impala.IMPALANetworks,
                               reverb.ReplaySample]
):
  """IMPALA Builder."""

  def __init__(
    self,
    config: impala.IMPALAConfig,
    core_state_spec: hk.LSTMState,
    num_envs: int,
    table_extension: Optional[Callable[[], Any]] = None,
  ):
    """Creates an IMPALA learner."""
    self._config = config
    self._core_state_spec = core_state_spec
    self._sequence_length = self._config.sequence_length
    self._num_sequences_per_batch = self._config.batch_size
    self._table_extension = table_extension
    self._num_envs = num_envs

  def make_replay_tables(
    self,
    environment_spec: specs.EnvironmentSpec,
    policy: impala.IMPALANetworks,
  ) -> List[reverb.Table]:
    """The queue; use XData or INFO log."""
    del policy
    num_actions = environment_spec.actions.num_values
    extra_spec = {
      "core_state": self._core_state_spec,
      "logits": jnp.ones(shape=(num_actions,), dtype=jnp.float32)
    }

    signature = BatchSequenceAdder.signature(
      environment_spec,
      extra_spec,
      sequence_length=self._sequence_length,
      batch_size=self._num_envs
    )

    # Maybe create rate limiter.
    # Setting the samples_per_insert ratio less than the default of 1.0, allows
    # the agent to drop data for the benefit of using data from most up-to-date
    # policies to compute its learner updates.
    samples_per_insert = self._config.samples_per_insert
    if samples_per_insert:
      if samples_per_insert > 1.0 or samples_per_insert <= 0.0:
        raise ValueError(
          "Impala requires a samples_per_insert ratio in the range (0, 1],"
          f" but received {samples_per_insert}."
        )
      limiter = reverb.rate_limiters.SampleToInsertRatio(
        samples_per_insert=samples_per_insert,
        min_size_to_sample=1,
        error_buffer=self._config.batch_size
      )
    else:
      limiter = reverb.rate_limiters.MinSize(1)

    table_extensions = []
    if self._table_extension is not None:
      table_extensions = [self._table_extension()]
    queue = reverb.Table(
      name=self._config.replay_table_name,
      sampler=reverb.selectors.Fifo(),
      remover=reverb.selectors.Fifo(),
      max_size=self._config.max_queue_size,
      max_times_sampled=1,
      rate_limiter=limiter,
      extensions=table_extensions,
      signature=signature
    )
    return [queue]

  def make_dataset_iterator(
    self, replay_client: reverb.Client
  ) -> Iterator[reverb.ReplaySample]:
    """Creates a dataset."""
    assert self._num_sequences_per_batch % self._num_envs == 0
    batch_size = self._num_sequences_per_batch // self._num_envs

    dataset = make_reverb_dataset(
      table=self._config.replay_table_name,
      server_address=replay_client.server_address,
      batch_size=batch_size,
      num_parallel_calls=None
    )

    def transpose(sample: reverb.ReplaySample) -> reverb.ReplaySample:
      _data = sample.data

      def _process(data):
        shape = data.shape
        data = tf.transpose(
          data, (0, 2, 1, *[i for i in range(3, len(shape))])
        )
        data = tf.reshape(
          data,
          (self._config.batch_size, self._config.sequence_length, *shape[3:])
        )
        return data

      return reverb.ReplaySample(
        info=tf.zeros([8]), data=tree.map_structure(_process, _data)
      )

    dataset = dataset.map(transpose)

    return dataset.as_numpy_iterator()

  def make_adder(self, replay_client: reverb.Client) -> Adder:
    """Creates an adder which handles observations."""
    # Note that the last transition in the sequence is used for bootstrapping
    # only and is ignored otherwise. So we need to make sure that sequences
    # overlap on one transition, thus "-1" in the period length computation.
    return AutoResetSequenceAdder(
      client=replay_client,
      sequence_length=self._sequence_length,
      period=self._config.sequence_period or (self._sequence_length - 1),
      priority_fns={self._config.replay_table_name: None},
    )

  def make_learner(
    self,
    random_key: networks_lib.PRNGKey,
    networks: impala.IMPALANetworks,
    dataset: Iterator[reverb.ReplaySample],
    logger: loggers.Logger,
    environment_spec: specs.EnvironmentSpec,
    replay_client: Optional[reverb.Client] = None,
    counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del environment_spec, replay_client

    optimizer = optax.chain(
      optax.clip_by_global_norm(self._config.max_gradient_norm),
      optax.adam(
        self._config.learning_rate,
        b1=self._config.adam_momentum_decay,
        b2=self._config.adam_variance_decay
      ),
    )

    return impala.IMPALALearner(
      networks=networks,
      iterator=dataset,
      optimizer=optimizer,
      random_key=random_key,
      discount=self._config.discount,
      entropy_cost=self._config.entropy_cost,
      baseline_cost=self._config.baseline_cost,
      max_abs_reward=self._config.max_abs_reward,
      counter=counter,
      logger=logger,
    )

  def make_actor(
    self,
    random_key: networks_lib.PRNGKey,
    policy: impala.IMPALANetworks,
    environment_spec: specs.EnvironmentSpec,
    variable_source: Optional[core.VariableSource] = None,
    adder: Optional[Adder] = None,
  ) -> acme.Actor:
    del environment_spec
    variable_client = variable_utils.VariableClient(
      client=variable_source, key="network", update_period=1000, device="cpu"
    )
    return IMPALAActor(
      forward_fn=policy.forward_fn,
      initial_state_fn=policy.initial_state_fn,
      batch_size=self._num_envs,
      variable_client=variable_client,
      adder=adder,
      rng=hk.PRNGSequence(random_key),
    )


class EnvironmentLoop(core.Worker):

  def __init__(
    self,
    environment: dm_env.Environment,
    actor: core.Actor,
    counter: Optional[counting.Counter] = None,
    logger: Optional[loggers.Logger] = None,
    label: str = "environment_loop",
  ) -> None:
    self._environment = environment
    self._actor = actor
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
      label, steps_key=self._counter.get_steps_key()
    )
    self._select_action = self._actor.select_action

  def run(self):
    episode_return = tree.map_structure(
      _generate_zeros_from_spec, self._environment.reward_spec()
    )
    episode_length = tree.map_structure(
      _generate_zeros_from_spec, self._environment.reward_spec()
    )
    timestep = self._environment.reset()
    print("reset")
    self._actor.observe_first(timestep)
    print("ob first")
    while True:
      print("step!")
      action = self._select_action(timestep.observation)
      action = utils.fetch_devicearray(action)
      timestep = self._environment.step(action)
      self._actor.observe(action, next_timestep=timestep)
      episode_return = tree.map_structure(
        operator.iadd, episode_return, timestep.reward
      )
      episode_length += 1
      for i, t in enumerate(timestep.step_type):
        if t == dm_env.StepType.LAST:
          result = {
            "episode_length": episode_length[i],
            "episode_return": episode_return[i],
          }
          self._logger.write(result)
          episode_length[i] = 0
          episode_return[i] = 0


class BatchEnvWrapper(dm_env.Environment):

  def __init__(
    self,
    environment: Union[DummyVecEnv, EnvPool],
    atari_normalize: bool = False,
    use_oar: bool = False,
  ):
    self._environment = environment
    if not isinstance(environment, DummyVecEnv):
      self._num_envs = len(environment.all_env_ids)
      self._use_env_pool = True
    else:
      self._num_envs = environment.num_envs
      self._use_env_pool = False
    self._reset_next_step = True
    self._atari_normalize = atari_normalize
    self._use_oar = use_oar

  def reset(self) -> TimeStep:
    self._reset_next_step = False
    observation = self._environment.reset()
    if self._atari_normalize:
      observation = observation.astype(np.float32) / 255
    ts = TimeStep(
      step_type=np.full(self._num_envs, dm_env.StepType.FIRST, dtype="int32"),
      reward=np.zeros(self._num_envs, dtype="float32"),
      discount=np.ones(self._num_envs, dtype="float32"),
      observation=observation
    )
    if self._use_oar:
      action = tree.map_structure(
        lambda x: x.generate_value(), self.action_spec()
      )
      reward = tree.map_structure(
        lambda x: x.generate_value(), self.reward_spec()
      )
      return self._augment_observation(action, reward, ts)
    else:
      return ts

  def step(self, action: types.NestedArray) -> TimeStep:
    if self._reset_next_step:
      return self.reset()
    if self._use_env_pool:
      observation, reward, done, _ = self._environment.step(action)
    else:
      self._environment.step_async(action)
      observation, reward, done, _ = self._environment.step_wait()
    self._reset_next_step = any(done)
    if self._atari_normalize:
      observation = observation.astype(np.float32) / 255
    ts = TimeStep(
      step_type=(done + 1).astype(np.int32),
      reward=reward,
      discount=(1 - done).astype(np.float32),
      observation=observation
    )
    if self._use_oar:
      return self._augment_observation(action, ts.reward, ts)
    else:
      return ts

  def observation_spec(self):
    space = self._environment.observation_space
    obs_spec = specs.BoundedArray(
      shape=space.shape,
      dtype="float32",
      minimum=space.low,
      maximum=space.high,
      name="observation"
    )
    if self._use_oar:
      return OAR(
        observation=obs_spec,
        # Use single action spec for parameter init for OAR.
        action=self.single_action_spec(),
        reward=self.single_reward_spec()
      )
    else:
      return obs_spec

  def single_action_spec(self):
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
      name="single_action"
    )

  def single_reward_spec(self):
    return specs.Array(
      name="single_reward",
      shape=[],
      dtype="float32",
    )

  def action_spec(self):
    spec = self.single_action_spec()
    if not self._atari_normalize:
      return spec
    spec._shape = (self._num_envs, *spec._shape)
    spec._name = "action"
    return spec

  def reward_spec(self):
    return specs.Array(
      name="reward",
      shape=[self._num_envs],
      dtype="float32",
    )

  def _augment_observation(
    self, action: types.NestedArray, reward: types.NestedArray,
    timestep: TimeStep
  ) -> TimeStep:
    oar = OAR(observation=timestep.observation, action=action, reward=reward)
    return timestep._replace(observation=oar)

  def close(self):
    self._environment.close()


class DeepIMPALAAtariNetwork(networks_lib.DeepIMPALAAtariNetwork):
  pass
  # def unroll(
  #   self, inputs: observation_action_reward.OAR, state: hk.LSTMState
  # ) -> networks_lib.LSTMOutputs:
  #   print(inputs.observation.shape)
  #   print(state.hidden.shape)
  #   embeddings = self._embed(inputs)
  #   if jnp.ndim(embeddings) == 2:
  #     # Add batch/time dimension.
  #     state = tree.map_structure(lambda x: jnp.expand_dims(x, 0), state)
  #   embeddings = jnp.expand_dims(embeddings, 1)
  #   embeddings, new_states = hk.static_unroll(
  #     self._core, embeddings, state, time_major=False
  #   )
  #   logits, values = self._head(embeddings)

  #   return (logits, values), new_states


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
  steps_key: str = "steps",
  task_instance: int = 0,
  run_name: str = "",
  wb_entity: str = "",
  config: dict = {},
) -> loggers.Logger:
  del task_instance, steps_key
  num_envs = config["num_envs"] if config["use_batch_env"] else 1

  print_fn = logging.info
  terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)
  loggers = [terminal_logger]

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
    loggers.append(WBLogger(num_envs))

  # Dispatch to all writers and filter Nones and by time.
  logger = aggregators.Dispatcher(loggers, base.to_numpy)
  logger = filters.NoneFilter(logger)

  return logger


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)
