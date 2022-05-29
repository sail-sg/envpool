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
"""Helpers for acme experiments."""

import logging
from functools import partial
from typing import Any, List, Mapping, Union

import dm_env
import gym
import jax
import jax.numpy as jnp
import numpy as np
import tree
from acme import types, wrappers
from acme.adders import Adder
from acme.agents.jax.actor_core import (
  ActorCore,
  FeedForwardPolicyWithExtra,
  SimpleActorCoreStateWithExtras,
)
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax.types import PRNGKey
from acme.utils.loggers import (
  Logger,
  aggregators,
  base,
  csv,
  filters,
  terminal,
)

import envpool
from envpool.python.protocol import EnvPool

Array = Union[np.ndarray, jnp.ndarray]


class TimeStep(dm_env.TimeStep):

  extras: Any

  def __new__(cls, step_type, reward, discount, observation, extras):
    self = super(TimeStep,
                 cls).__new__(cls, step_type, reward, discount, observation)
    self.extras = extras
    return self


class EnvPoolWrapper(wrappers.EnvironmentWrapper):

  def __init__(self, environment: EnvPool):
    super().__init__(environment)
    self._num_envs = len(environment.all_env_ids)
    self._is_done = False

  def observation_spec(self) -> Array:
    obs = self._environment.observation_spec().obs
    new_obs = dm_env.specs.BoundedArray(
      name=obs.name,
      shape=[s for s in obs.shape if s != -1],
      dtype="float32",
      minimum=obs.minimum,
      maximum=obs.maximum,
    )
    return new_obs

  def reset(self) -> dm_env.TimeStep:
    self._is_done = False
    ts = super().reset()
    return dm_env.TimeStep(
      step_type=dm_env.StepType.FIRST,
      observation=ts.observation.obs,
      reward=ts.reward,
      discount=ts.discount,
    )

  def step(self, action) -> TimeStep:
    ts = super().step(action)
    _is_done = np.sum(ts.step_type == dm_env.StepType.LAST)
    if _is_done > 0:
      assert _is_done == self._num_envs, "envs do not finish at the same time."
      self._is_done = True
    ts = TimeStep(
      step_type=dm_env.StepType.LAST if self._is_done else dm_env.StepType.MID,
      observation=ts.observation.obs,
      reward=ts.reward[0],  # Single value for recording the return.
      discount=ts.discount,
      extras={
        "step_type": ts.step_type,
        "reward": ts.reward,
      }
    )
    return ts


class AdderWrapper(Adder):

  def __init__(self, adders: List[Adder]) -> None:
    self._adders: List[Adder] = adders

  def reset(self):
    for adder in self._adders:
      adder.reset()

  def add_first(self, timestep: dm_env.TimeStep):
    for i, adder in enumerate(self._adders):
      ts = dm_env.TimeStep(
        step_type=timestep.step_type,
        observation=timestep.observation[i],
        reward=timestep.reward[i],
        discount=timestep.discount[i],
      )
      adder.add_first(ts)

  def add(
    self,
    action: types.NestedArray,
    next_timestep: TimeStep,
    extras: types.NestedArray = ...
  ):
    for i, adder in enumerate(self._adders):
      timestep = dm_env.TimeStep(
        step_type=next_timestep.extras["step_type"][i],
        observation=next_timestep.observation[i],
        reward=next_timestep.extras["reward"][i],
        discount=next_timestep.discount[i],
      )
      if extras is not None:
        _extras = tree.map_structure(lambda x: utils.to_numpy(x[i]), extras)
      adder.add(action[i], timestep, _extras)


def batched_feed_forward_with_extras_to_actor_core(
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


def make_environment(task: str, use_envpool: bool = False, num_envs: int = 2):
  env_wrappers = []
  if use_envpool:
    env = envpool.make(
      task,
      env_type="dm",
      num_envs=num_envs,
    )
  else:
    env = gym.make(task)
    # Make sure the environment obeys the dm_env.Environment interface.
    env_wrappers.append(wrappers.GymWrapper)
  # Clip the action returned by the agent to the environment spec.
  env_wrappers += [
    partial(wrappers.CanonicalSpecWrapper, clip=True),
    wrappers.SinglePrecisionWrapper
  ]
  if use_envpool:
    env_wrappers.append(EnvPoolWrapper)
  return wrappers.wrap_all(env, env_wrappers)


def make_logger(
  label: str,
  steps_key: str = "steps",
  task_instance: int = 0,
  run_name: str = "",
  wb_entity: str = "",
) -> Logger:

  import wandb

  del steps_key, task_instance
  print_fn = logging.info
  terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)
  loggers = [terminal_logger]
  loggers.append(csv.CSVLogger(label=label))

  class WBLogger(base.Logger):

    def __init__(self, label: str = "") -> None:
      super().__init__()
      wandb.init(
        project="EnvPool",
        entity=wb_entity,
        name=run_name,
        job_type=label,
      )

    def write(self, data: base.LoggingData) -> None:
      data = base.to_numpy(data)
      wandb.log(data)

    def close(self) -> None:
      wandb.finish()

  if label == "train":
    loggers.append(WBLogger(label=label))

  # Dispatch to all writers and filter Nones and by time.
  logger = aggregators.Dispatcher(loggers, base.to_numpy)
  logger = filters.NoneFilter(logger)
  logger = filters.TimeFilter(logger, 0.2)

  return logger
