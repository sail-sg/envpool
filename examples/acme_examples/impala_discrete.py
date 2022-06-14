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
"""Example running IMPALA on Atari games tasks using EnvPool."""

import argparse
import dataclasses
import logging
import os
from dataclasses import asdict
from functools import partial
from typing import Any, Generic, Optional, Tuple

import acme
import dm_env
import gym
import haiku as hk
import jax
import jax.numpy as jnp
import launchpad as lp
import numpy as np
import optax
import reverb
import tensorflow as tf
import tree
from acme import adders, core, specs, types, wrappers
from acme.adders import Adder
from acme.adders import reverb as reverb_adders
from acme.adders.reverb import base as reverb_base
from acme.adders.reverb import sequence as reverb_sequence
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors, builders, impala, ppo
from acme.agents.jax.actor_core import (
  ActorCore,
  FeedForwardPolicyWithExtra,
  SimpleActorCoreStateWithExtras,
)
from acme.agents.jax.impala import acting
from acme.agents.jax.impala import config as impala_config
from acme.agents.jax.impala import learning
from acme.agents.jax.impala import networks as impala_networks
from acme.agents.jax.impala import types as impala_types
from acme.agents.jax.impala.networks import HaikuLSTMOutputs
from acme.datasets import reverb as datasets
from acme.jax import experiments
from acme.jax import networks as networks_lib
from acme.jax import savers, utils, variable_utils
from acme.jax.experiments import CheckpointingConfig
from acme.jax.types import PRNGKey
from acme.utils import counting, experiment_utils, loggers
from acme.utils.loggers import Logger, aggregators, base, filters, terminal
from acme.utils.lp_utils import StepsLimiter
from acme_envpool_utils.helpers import (
  AdderWrapper,
  AutoResetSequenceAdder,
  BatchEnvWrapper,
  BatchSequenceAdder,
  DeepIMPALAAtariNetwork,
  EnvironmentLoop,
  IMPALABuilder,
  make_logger,
)
from acme_envpool_utils.lp_utils import run_distributed_experiment
from dm_env import StepType
from importlib_metadata import itertools
from stable_baselines3.common.env_util import make_vec_env

import envpool
from envpool.python.protocol import EnvPool

logging.getLogger().setLevel(logging.INFO)


def make_atari_environment(
  task: str, use_envpool: bool = False, num_envs: int = 2
):
  env_wrappers = []
  if use_envpool:
    env = envpool.make(
      task,
      env_type="gym",
      num_envs=num_envs,
      episodic_life=True,
      reward_clip=True
    )
    env_wrappers.append(
      partial(BatchEnvWrapper, atari_normalize=True, use_oar=True)
    )
  else:
    raise
  # env_wrappers.append(wrappers.ObservationActionRewardWrapper)
  env_wrappers.append(wrappers.SinglePrecisionWrapper)
  return wrappers.wrap_all(env, env_wrappers)


# while all(ts.discount): ts = env.step(np.array([0]*8))


def build_distributed_program(FLAGS):
  import pdb

  # pdb.set_trace()

  task = FLAGS.env_name

  use_envpool = FLAGS.use_envpool

  num_envs = FLAGS.num_envs if use_envpool else -1
  num_steps = FLAGS.num_steps // FLAGS.num_envs if \
     use_envpool else FLAGS.num_steps

  config = impala.IMPALAConfig(max_queue_size=int(1e6))

  if FLAGS.use_wb:
    run_name = f"acme_ppo__{FLAGS.env_name}"
    if FLAGS.use_envpool:
      run_name += f"__envpool-{FLAGS.num_envs}"
    elif FLAGS.use_vec_env:
      run_name += f"__vec_env-{FLAGS.num_envs}"
    run_name += f"__seed-{FLAGS.seed}"
    config.update(vars(FLAGS))
    logger_factory = partial(
      make_logger, run_name=run_name, wb_entity=FLAGS.wb_entity, config=config
    )
  else:
    logger_factory = experiment_utils.make_experiment_logger
  core_state_spec = hk.LSTMState(jnp.zeros([256]), cell=jnp.zeros([256]))
  impala_builder = IMPALABuilder(config, core_state_spec, num_envs)

  config = asdict(config)

  def environment_factory(_):
    return make_atari_environment(task, use_envpool, num_envs)

  impala_builder.make_replay_tables(
    specs.make_environment_spec(environment_factory(0)), None
  )

  checkpointing_config = CheckpointingConfig()

  def build_replay():
    spec = specs.make_environment_spec(environment_factory(0))
    return impala_builder.make_replay_tables(spec, None)

  def build_counter():
    return savers.CheckpointingRunner(
      counting.Counter(),
      key="counter",
      subdirectory="counter",
      time_delta_minutes=5,
      directory=checkpointing_config.directory,
      add_uid=checkpointing_config.add_uid,
      max_to_keep=checkpointing_config.max_to_keep
    )

  def build_learner(
    random_key: networks_lib.PRNGKey,
    replay: reverb.Client,
    counter: Optional[counting.Counter] = None,
  ):
    spec = specs.make_environment_spec(environment_factory(0))
    networks = make_atari_networks(spec, num_envs if num_envs > 0 else None)
    iterator = impala_builder.make_dataset_iterator(replay)
    logger = logger_factory("learner", "learner_steps", 0)
    counter = counting.Counter(counter, "learner")
    learner = impala_builder.make_learner(
      random_key, networks, iterator, logger, spec, replay, counter
    )
    learner = savers.CheckpointingRunner(
      learner,
      key="learner",
      subdirectory="learner",
      time_delta_minutes=5,
      directory=checkpointing_config.directory,
      add_uid=checkpointing_config.add_uid,
      max_to_keep=checkpointing_config.max_to_keep
    )
    return learner

  def build_actor(
    random_key: networks_lib.PRNGKey,
    replay: reverb.Client,
    variable_source: core.VariableSource,
    counter: counting.Counter,
    actor_id: int,
  ) -> EnvironmentLoop:
    adder = impala_builder.make_adder(replay)
    environment = environment_factory(0)
    environment_spec = specs.make_environment_spec(environment)
    networks = make_atari_networks(
      environment_spec, num_envs if num_envs > 0 else None
    )
    actor = impala_builder.make_actor(
      random_key, networks, environment_spec, variable_source, adder
    )
    counter = counting.Counter(counter, "actor")
    logger = logger_factory("actor", "actor_steps", actor_id)
    return EnvironmentLoop(environment, actor, counter, logger)

  # program = lp.Program(name="impala_agent")
  counter = build_counter()
  key = jax.random.PRNGKey(FLAGS.seed)
  replay_tables = build_replay()
  replay_server = reverb.Server(replay_tables, port=None)
  replay_client = reverb.Client(f'localhost:{replay_server.port}')
  learner = build_learner(key, replay_client, counter)

  actor = build_actor(key, replay_client, learner, counter, 0)
  actor.run()

  # replay_node = lp.ReverbNode(
  #   build_replay,
  #   checkpoint_time_delta_minutes=(
  #     checkpointing_config.replay_checkpointing_time_delta_minutes
  #   )
  # )
  # replay = replay_node.create_handle()

  # counter = program.add_node(lp.CourierNode(build_counter), label="counter")

  # program.add_node(
  #   lp.CourierNode(StepsLimiter, counter, num_steps), label="counter"
  # )

  # learner_key, key = jax.random.split(key)
  # learner_node = lp.CourierNode(build_learner, learner_key, replay, counter)
  # learner = learner_node.create_handle()

  # variable_sources = [learner]
  # program.add_node(replay_node, label="replay")

  # with program.group("actor"):
  #   *actor_keys, key = jax.random.split(key, FLAGS.num_actors + 1)
  #   variable_sources = itertools.cycle(variable_sources)
  #   actor_nodes = [
  #     lp.CourierNode(build_actor, akey, replay, vsource, counter, aid)
  #     for aid, (akey, vsource) in enumerate(zip(actor_keys, variable_sources))
  #   ]
  #   for actor_node in actor_nodes:
  #     program.add_node(actor_node)

  # return program


def main():
  FLAGS = parse_args()
  program = build_distributed_program(FLAGS)

  # run_distributed_experiment(program, None, None)


def parse_args():
  # fmt: off
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--exp-name",
    type=str,
    default=os.path.basename(__file__).rstrip(".py"),
    help="Name of this experiment."
  )
  parser.add_argument(
    "--env-name", type=str, default="Pong-v5", help="What environment to run."
  )
  parser.add_argument(
    "--use-envpool",
    type=bool,
    default=False,
    nargs="?",
    const=True,
    help="Whether to use EnvPool."
  )
  parser.add_argument(
    "--num-envs",
    type=int,
    default=8,
    help="Number of environments (EnvPool)."
  )
  parser.add_argument(
    "--num-actors", type=int, default=1, help="Number of actors (LaunchPad)."
  )
  parser.add_argument("--seed", type=int, default=0, help="Random seed.")
  parser.add_argument(
    "--num-steps",
    type=int,
    default=1_000_000,
    help="Number of env steps to run."
  )
  parser.add_argument(
    "--use-wb",
    type=bool,
    default=False,
    nargs="?",
    const=True,
    help="Whether to use WandB."
  )
  parser.add_argument(
    "--wb-project", type=str, default="acme", help="W&B project name."
  )
  parser.add_argument(
    "--wb-entity", type=str, default=None, help="W&B entity name."
  )
  args = parser.parse_args()
  return args


def make_haiku_networks(
  env_spec: specs.EnvironmentSpec,
  forward_fn: Any,
  initial_state_fn: Any,
  unroll_fn: Any,
  batch_size: Optional[int] = None,
) -> impala.IMPALANetworks[impala_types.RecurrentState]:
  """Builds functional impala network from recurrent model definitions."""
  # Make networks purely functional.
  forward_hk = hk.without_apply_rng(hk.transform(forward_fn))
  initial_state_hk = hk.without_apply_rng(hk.transform(initial_state_fn))
  unroll_hk = hk.without_apply_rng(hk.transform(unroll_fn))

  # Note: batch axis is not needed for the actors.
  dummy_obs = utils.zeros_like(env_spec.observations)
  dummy_obs_sequence = utils.add_batch_dim(dummy_obs)

  def unroll_init_fn(
    rng: networks_lib.PRNGKey, initial_state: impala_types.RecurrentState
  ) -> hk.Params:
    print(tree.map_structure(lambda x: x.shape, dummy_obs_sequence))
    return unroll_hk.init(rng, dummy_obs_sequence, initial_state)

  return impala.IMPALANetworks(
    forward_fn=forward_hk.apply,
    unroll_init_fn=unroll_init_fn,
    unroll_fn=unroll_hk.apply,
    initial_state_fn=(
      lambda rng: initial_state_hk.apply(initial_state_hk.init(rng))
    )
  )


def make_atari_networks(
  env_spec: specs.EnvironmentSpec,
  batch_size: Optional[int] = None,
) -> impala.IMPALANetworks[hk.LSTMState]:
  """Builds default IMPALA networks for Atari games."""

  def forward_fn(
    inputs: impala_types.Observation, state: hk.LSTMState
  ) -> HaikuLSTMOutputs:
    model = DeepIMPALAAtariNetwork(env_spec.actions.num_values)
    return model(inputs, state)

  def initial_state_fn() -> hk.LSTMState:
    model = DeepIMPALAAtariNetwork(env_spec.actions.num_values)
    return model.initial_state(None)

  def unroll_fn(
    inputs: impala_types.Observation, state: hk.LSTMState
  ) -> HaikuLSTMOutputs:
    model = DeepIMPALAAtariNetwork(env_spec.actions.num_values)
    return model.unroll(inputs, state)

  return make_haiku_networks(
    env_spec=env_spec,
    forward_fn=forward_fn,
    initial_state_fn=initial_state_fn,
    unroll_fn=unroll_fn,
    batch_size=batch_size,
  )


if __name__ == "__main__":
  main()
