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
"""Example running PPO on mujoco tasks.

Acme has only released v0.4.0 on PyPI for now (22/05/29), which is far behind
the master codes, where APIs for constructing experiments were added.

We are using the newest master version (344022e), so please make sure you
install acme using method 4 (https://github.com/deepmind/acme#installation).
"""

import time
from functools import partial
from typing import Any, List, Mapping, Optional, Union

import dm_env
import gym
import jax
import jax.numpy as jnp
import numpy as np
import reverb
import tree
from absl import app, flags
from acme import core, types, wrappers
from acme.adders import Adder
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors, ppo
from acme.agents.jax.actor_core import (
  ActorCore,
  FeedForwardPolicyWithExtra,
  SimpleActorCoreStateWithExtras,
)
from acme.jax import experiments
from acme.jax import networks as networks_lib
from acme.jax import utils, variable_utils
from acme.jax.types import PRNGKey
from acme.utils import loggers

import envpool
from envpool.python.protocol import EnvPool

Array = Union[np.ndarray, jnp.ndarray]

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "HalfCheetah-v3", "What environment to run.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("num_steps", 1_000_000, "Number of env steps to run.")
flags.DEFINE_integer("eval_every", 50_000, "How often to run evaluation.")
flags.DEFINE_boolean("use_envpool", False, "Whether to use EnvPool.")
flags.DEFINE_integer("num_envs", 8, "Number of environment.")
flags.DEFINE_boolean("use_wb", False, "Whether to use WandB.")
flags.DEFINE_string("wb_entity", None, "WandB entity name.")


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
      reward=np.sum(ts.reward),  # Single value for recording the return.
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


class BuilderWrapper(ppo.PPOBuilder):

  def __init__(self, config: ppo.PPOConfig, num_envs: int = -1):
    super().__init__(config)
    self._num_envs = num_envs
    self._use_envpool = num_envs > 0

  def make_adder(self, replay_client: reverb.Client) -> Adder:
    if not self._use_envpool:
      return super(BuilderWrapper, self).make_adder(replay_client)
    return AdderWrapper(
      [
        super(BuilderWrapper, self).make_adder(replay_client)
        for _ in range(self._num_envs)
      ]
    )

  def make_actor(
    self,
    random_key: networks_lib.PRNGKey,
    policy_network: actor_core_lib.FeedForwardPolicyWithExtra,
    adder: Optional[Adder] = None,
    variable_source: Optional[core.VariableSource] = None
  ) -> core.Actor:
    if not self._use_envpool:
      return super().make_actor(
        random_key, policy_network, adder, variable_source
      )
    assert variable_source is not None
    variable_client = variable_utils.VariableClient(
      variable_source,
      'network',
      device='cpu',
      update_period=self._config.variable_update_period
    )
    actor = batched_feed_forward_with_extras_to_actor_core(policy_network)
    return actors.GenericActor(
      actor, random_key, variable_client, adder, backend='cpu'
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


def build_experiment_config():
  task = FLAGS.env_name
  use_envpool = FLAGS.use_envpool
  num_envs = FLAGS.num_envs if use_envpool else -1
  num_steps = FLAGS.num_steps // FLAGS.num_envs if \
     FLAGS.use_envpool else FLAGS.num_steps

  config = ppo.PPOConfig(entropy_cost=0, learning_rate=1e-4)

  ppo_builder = BuilderWrapper(config, num_envs)

  layer_sizes = (256, 256, 256)

  return experiments.Config(
    builder=ppo_builder,
    environment_factory=lambda _: make_environment(
      task,
      use_envpool=use_envpool,
      num_envs=num_envs,
    ),
    network_factory=lambda spec: ppo.make_networks(spec, layer_sizes),
    policy_network_factory=ppo.make_inference_fn,
    seed=FLAGS.seed,
    max_number_of_steps=num_steps
  )


def make_logger(
  label: str,
  steps_key: str = 'steps',
  task_instance: int = 0,
) -> loggers.Logger:

  import logging

  from acme.utils.loggers import aggregators
  from acme.utils.loggers import asynchronous as async_logger
  from acme.utils.loggers import base, csv, filters, terminal

  import wandb

  del steps_key, task_instance
  print_fn = logging.info
  terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)
  loggers = [terminal_logger]
  loggers.append(csv.CSVLogger(label=label))

  run_name = f"acme_ppo__{FLAGS.env_name}"
  if FLAGS.use_envpool:
    run_name += "__envpool"
  run_name += f"__{FLAGS.seed}__{int(time.time())}"

  class WBLogger(base.Logger):

    def __init__(self, label: str = "") -> None:
      super().__init__()
      wandb.init(
        project="EnvPool",
        entity=FLAGS.wb_entity,
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


def main(_):
  config = build_experiment_config()
  if FLAGS.use_wb:
    config.logger_factory = make_logger
  # If we enable envpool, we're evaluating on
  # (num_eval_episodes * num_envs) episodes.
  experiments.run_experiment(
    experiment=config, eval_every=FLAGS.eval_every, num_eval_episodes=10
  )


if __name__ == "__main__":
  app.run(main)
