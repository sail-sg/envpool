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
from dataclasses import asdict
from functools import partial
from typing import Iterable, Iterator, List, Optional

import acme_envpool_helpers.helpers as helpers
import acme_envpool_helpers.lp_helpers as lp_helpers
import jax
import numpy as np
import reverb
import tensorflow as tf
import tree
from absl import app, flags
from acme import core, specs, types
from acme.adders import Adder
from acme.adders.reverb import base as reverb_base
from acme.adders.reverb import sequence as reverb_sequence
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors, ppo
from acme.jax import experiments
from acme.jax import networks as networks_lib
from acme.jax import utils, variable_utils

FLAGS = flags.FLAGS

flags.DEFINE_bool(
  "run_distributed", False, "Should an agent be executed in a "
  "distributed way (the default is a single-threaded agent)"
)
flags.DEFINE_integer("num_actors", 1, "Number of actors.")
flags.DEFINE_string("env_name", "HalfCheetah-v3", "What environment to run.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("num_steps", 1_000_000, "Number of env steps to run.")
flags.DEFINE_boolean("use_envpool", False, "Whether to use EnvPool.")
flags.DEFINE_integer("num_envs", 8, "Number of environment.")
flags.DEFINE_boolean("use_wb", False, "Whether to use WandB.")
flags.DEFINE_string("wb_entity", None, "WandB entity name.")
flags.DEFINE_string("desc", "", "More description for the run.")


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
      return tf.TensorSpec(
        shape=(sequence_length, batch_size, *spec.shape),
        dtype=spec.dtype,
        name='/'.join(str(p) for p in paths)
      )

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


class BuilderWrapper(ppo.PPOBuilder):
  """Wrap the PPO algorithm builder for EnvPool."""

  def __init__(self, config: ppo.PPOConfig, num_envs: int = -1):
    super().__init__(config)
    self._num_envs = num_envs
    self._use_envpool = num_envs > 0

  def make_replay_tables(
    self, environment_spec: specs.EnvironmentSpec
  ) -> List[reverb.Table]:
    if not self._use_envpool:
      return super(BuilderWrapper, self).make_replay_tables(environment_spec)
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
        max_size=self._config.batch_size,
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
    return helpers.AdderWrapper(adder)

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
      "network",
      device="cpu",
      update_period=self._config.variable_update_period
    )
    actor = helpers.batched_feed_forward_with_extras_to_actor_core(
      policy_network
    )
    return actors.GenericActor(
      actor, random_key, variable_client, adder, backend="cpu"
    )


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
    environment_factory=lambda _: helpers.make_environment(
      task,
      use_envpool=use_envpool,
      num_envs=num_envs,
    ),
    network_factory=lambda spec: ppo.make_networks(spec, layer_sizes),
    policy_network_factory=ppo.make_inference_fn,
    evaluator_factories=[],
    seed=FLAGS.seed,
    max_number_of_steps=num_steps - 1
  ), config


def main(_):
  experiment, config = build_experiment_config()
  if FLAGS.use_wb:
    run_name = f"ppo__{FLAGS.env_name}"
    if FLAGS.use_envpool:
      run_name += f"__envpool-{FLAGS.num_envs}"
    if FLAGS.run_distributed:
      run_name += f"__dist-{FLAGS.num_actors}"
    if FLAGS.desc:
      run_name += f"__{FLAGS.desc}"
    run_name += f"__{FLAGS.seed}__{int(time.time())}"
    cfg = asdict(config)
    cfg.update(FLAGS.flag_values_dict().items())
    experiment.logger_factory = partial(
      helpers.make_logger,
      run_name=run_name,
      wb_entity=FLAGS.wb_entity,
      config=cfg
    )

  if FLAGS.run_distributed:
    lp_helpers.run_distributed_experiment(
      experiment=experiment, num_actors=FLAGS.num_actors
    )
  else:
    experiments.run_experiment(
      experiment=experiment,
      eval_every=experiment.max_number_of_steps,
    )


if __name__ == "__main__":
  app.run(main)
