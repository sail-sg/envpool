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
"""Example running PPO on mujoco tasks."""

import argparse
import collections
import logging
import os
from dataclasses import asdict
from functools import partial
from typing import Iterable, Iterator, List, Mapping, Optional

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
from acme.jax import experiments
from acme.jax import networks as networks_lib
from acme.jax import utils, variable_utils
from acme.jax.types import PRNGKey
from acme.utils.loggers import Logger, aggregators, base, filters, terminal
from dm_env import StepType

import envpool
from envpool.python.protocol import EnvPool

logging.getLogger().setLevel(logging.INFO)


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
    "--env-name",
    type=str,
    default="HalfCheetah-v3",
    help="What environment to run."
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
    "--use-vec-env",
    type=bool,
    default=False,
    nargs="?",
    const=True,
    help="Whether to use SB3 VecEnv."
  )
  parser.add_argument(
    "--num-envs",
    type=int,
    default=8,
    help="Number of environment (EnvPool) / actor (LaunchPad)."
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


class TimeStep(dm_env.TimeStep):

  def last(self) -> bool:
    """Adapt the batch step_type for EnvironmentLoop episode-end checking."""
    return any(self.step_type == StepType.LAST)


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
      name = '/'.join(str(p) for p in paths)
      if "reward" in name:
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
        name='start_of_episode'
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
        'adder.add_first with an initial timestep (i.e. one for '
        'which timestep.first() is True'
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


class BuilderWrapper(ppo.PPOBuilder):
  """Wrap the PPO algorithm builder for EnvPool."""

  def __init__(self, config: ppo.PPOConfig, num_envs: int = -1):
    super().__init__(config)
    self._num_envs = num_envs
    self._use_envpool = num_envs > 0

  def make_replay_tables(
    self, environment_spec: specs.EnvironmentSpec,
    policy: actor_core_lib.FeedForwardPolicyWithExtra
  ) -> List[reverb.Table]:
    if not self._use_envpool:
      return super(BuilderWrapper,
                   self).make_replay_tables(environment_spec, policy)
    extra_spec = {
      'log_prob': np.ones(shape=(), dtype=np.float32),
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
    if not self._use_envpool:
      return super(BuilderWrapper, self).make_dataset_iterator(replay_client)
    assert self._config.batch_size % self._num_envs == 0
    batch_size = self._config.batch_size // self._num_envs
    dataset = reverb.TrajectoryDataset.from_table_signature(
      server_address=replay_client.server_address,
      table=self._config.replay_table_name,
      max_in_flight_samples_per_worker=2 * batch_size
    )

    def transpose(sample: reverb.ReplaySample) -> reverb.ReplaySample:
      data = sample.data

      def _process_rank_4(data):
        return tf.reshape(
          tf.transpose(data, [0, 2, 1, 3]),
          [self._config.batch_size, self._config.unroll_length + 1, -1]
        )

      def _process_rank_3(data):
        return tf.reshape(
          tf.transpose(data, [0, 2, 1]),
          [self._config.batch_size, self._config.unroll_length + 1]
        )

      reward = _process_rank_3(data.reward)
      return reverb.ReplaySample(
        info=sample.info,
        data=types.Transition(
          observation=_process_rank_4(data.observation),
          action=_process_rank_4(data.action),
          reward=reward,
          discount=_process_rank_3(data.discount),
          next_observation=tf.zeros_like(reward),
          extras={"log_prob": _process_rank_3(data.extras["log_prob"])}
        ),
      )

    # Add batch dimension.
    dataset = dataset.batch(batch_size, drop_remainder=True).map(transpose)
    return utils.device_put(dataset.as_numpy_iterator(), jax.devices()[0])

  def make_adder(self, replay_client: reverb.Client) -> Adder:
    adder = super(BuilderWrapper, self).make_adder(replay_client)
    if not self._use_envpool:
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
    if not self._use_envpool:
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


class EnvPoolWrapper(wrappers.EnvironmentWrapper):

  def __init__(self, environment: EnvPool):
    super().__init__(environment)
    self._num_envs = len(environment.all_env_ids)

  def observation_spec(self) -> dm_env.specs.BoundedArray:
    obs = self._environment.observation_spec().obs
    new_obs = dm_env.specs.BoundedArray(
      name=obs.name,
      shape=[s for s in obs.shape if s != -1],
      dtype="float32",
      minimum=obs.minimum,
      maximum=obs.maximum,
    )
    return new_obs

  def reward_spec(self):
    rew = self._environment.reward_spec()
    new_rew = dm_env.specs.Array(
      name=rew.name,
      shape=[self._num_envs],
      dtype="float32",
    )
    return new_rew

  def reset(self) -> TimeStep:
    ts = super().reset()
    return TimeStep(
      step_type=ts.step_type,
      observation=ts.observation.obs,
      reward=ts.reward,
      discount=ts.discount,
    )

  def step(self, action) -> TimeStep:
    ts = super().step(action)
    ts = TimeStep(
      step_type=ts.step_type,
      observation=ts.observation.obs,
      reward=ts.reward,
      # reward=ts.reward[0],  # Single value for recording the return.
      discount=ts.discount,
    )
    return ts


def make_environment(
  task: str, use_envpool: bool = False, use_vec_env=False, num_envs: int = 2
):
  env_wrappers = []
  if use_envpool:
    env = envpool.make(
      task,
      env_type="dm",
      num_envs=num_envs,
    )
  elif use_vec_env:
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
    env = make_vec_env(task, n_envs=num_envs)

    class SB3VecEnvWrapper(dm_env.Environment):

      def __init__(self, environment: DummyVecEnv):

        self._environment = environment
        self._num_envs = environment.num_envs
        self._reset_next_step = True

      def reset(self) -> dm_env.TimeStep:
        self._reset_next_step = False
        observation = self._environment.reset()
        return dm_env.TimeStep(
          step_type=np.full(self._num_envs, StepType.FIRST, dtype="int32"),
          reward=np.zeros(self._num_envs, dtype="float32"),
          discount=np.ones(self._num_envs, dtype="float32"),
          observation=observation
        )

      def step(self, action: types.NestedArray) -> dm_env.TimeStep:
        if self._reset_next_step:
          return self.reset()
        observation, reward, done, _ = self._environment.step(action)
        self._reset_next_step = any(done)
        return dm_env.TimeStep(
          step_type=(done + 1).astype(np.int32),
          reward=reward,
          discount=(1 - done).astype(np.float32),
          observation=observation
        )

      def observation_spec(self):
        return super().observation_spec()

      def action_spec(self):
        return super().action_spec()

      def close(self):
        self._environment.close()
  else:
    env = gym.make(task)
    # Make sure the environment obeys the dm_env.Environment interface.
    env_wrappers.append(wrappers.GymWrapper)
    # Clip the action returned by the agent to the environment spec.
    env_wrappers.append(partial(wrappers.CanonicalSpecWrapper, clip=True))
  env_wrappers.append(wrappers.SinglePrecisionWrapper)
  if use_envpool:
    env_wrappers.append(EnvPoolWrapper)
  return wrappers.wrap_all(env, env_wrappers)


def make_logger(
  label: str,
  steps_key: str = "steps",
  task_instance: int = 0,
  run_name: str = "",
  wb_entity: str = "",
  config: dict = {},
) -> Logger:
  del task_instance, steps_key
  num_envs = config["num_envs"] if config["use_envpool"] else 1

  print_fn = logging.info
  terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)
  loggers = [terminal_logger]

  if label == "train":

    import wandb

    class WBLogger(Logger):

      def __init__(self, num_envs) -> None:
        self._num_envs = num_envs
        self._avg_return = collections.deque(maxlen=20 * num_envs)

      def write(self, data) -> None:
        new_data = {}
        for key, value in data.items():
          if key in ["train_steps", "actor_steps"]:
            key = "global_step"
            value *= self._num_envs
          elif key == "episode_return":
            if self._num_envs > 1:
              self._avg_return.extend(value)
            else:
              self._avg_return.append(value)
            avg_return = np.array(self._avg_return)[::self._num_envs]
            value = np.average(avg_return)
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


def build_experiment_config(FLAGS):
  task = FLAGS.env_name
  use_envpool = FLAGS.use_envpool
  use_vec_env = not use_envpool and FLAGS.use_vec_env
  num_envs = FLAGS.num_envs if use_envpool else -1
  num_steps = FLAGS.num_steps // FLAGS.num_envs if \
     FLAGS.use_envpool else FLAGS.num_steps

  config = ppo.PPOConfig()
  ppo_builder = BuilderWrapper(config, num_envs)

  layer_sizes = (256, 256, 256)

  return experiments.Config(
    builder=ppo_builder,
    environment_factory=lambda _: make_environment(
      task,
      use_envpool=use_envpool,
      use_vec_env=use_vec_env,
      num_envs=num_envs,
    ),
    network_factory=lambda spec: ppo.make_networks(spec, layer_sizes),
    policy_network_factory=ppo.make_inference_fn,
    evaluator_factories=[],
    seed=0,
    max_number_of_steps=num_steps - 1
  ), config


def main():
  logging.info(f"Jax Devices: {jax.devices()}")
  FLAGS = parse_args()
  experiment, config = build_experiment_config(FLAGS)
  if FLAGS.use_wb:
    run_name = f"acme_ppo__{FLAGS.env_name}"
    if FLAGS.use_envpool:
      run_name += f"__envpool-{FLAGS.num_envs}"
    run_name += f"__seed-{FLAGS.seed}"
    cfg = asdict(config)
    cfg.update(vars(FLAGS))
    experiment.logger_factory = partial(
      make_logger, run_name=run_name, wb_entity=FLAGS.wb_entity, config=cfg
    )

  experiments.run_experiment(
    experiment=experiment,
    eval_every=experiment.max_number_of_steps,
  )


if __name__ == "__main__":
  main()
