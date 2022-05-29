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
"""Example running SAC on mujoco tasks.

Acme has only released v0.4.0 on PyPI for now (22/05/29), which is far behind
the master codes, where APIs for constructing experiments were added.

We are using the newest master version (344022e), so please make sure you
install acme using method 4 (https://github.com/deepmind/acme#installation).
"""

import time
from functools import partial
from typing import Optional

import helpers
import launchpad as lp
import reverb
from absl import app, flags
from acme import core, specs
from acme.adders import Adder
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors, normalization, ppo, sac
from acme.agents.jax.sac import builder
from acme.jax import experiments
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.utils import lp_utils

FLAGS = flags.FLAGS

flags.DEFINE_bool(
  "run_distributed", False, "Should an agent be executed in a "
  "distributed way (the default is a single-threaded agent)"
)
flags.DEFINE_integer("num_actors", 1, "Number of actors.")
flags.DEFINE_string("env_name", "HalfCheetah-v3", "What environment to run.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("num_steps", 1_000_000, "Number of env steps to run.")
flags.DEFINE_integer("eval_every", 50_000, "How often to run evaluation.")
flags.DEFINE_boolean("use_envpool", False, "Whether to use EnvPool.")
flags.DEFINE_integer("num_envs", 8, "Number of environment.")
flags.DEFINE_boolean("use_wb", False, "Whether to use WandB.")
flags.DEFINE_string("wb_entity", None, "WandB entity name.")


class BuilderWrapper(builder.SACBuilder):

  def __init__(self, config: sac.SACConfig, num_envs: int = -1):
    super().__init__(config)
    self._num_envs = num_envs
    self._use_envpool = num_envs > 0

  def make_adder(self, replay_client: reverb.Client) -> Adder:
    if not self._use_envpool:
      return super(BuilderWrapper, self).make_adder(replay_client)
    return helpers.AdderWrapper(
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
      "policy",
      device="cpu",
    )
    actor = helpers.batched_feed_forward_to_actor_core(policy_network)
    return actors.GenericActor(
      actor, random_key, variable_client, adder, backend="cpu"
    )


def build_experiment_config():
  task = FLAGS.env_name
  use_envpool = FLAGS.use_envpool
  num_envs = FLAGS.num_envs if use_envpool else -1
  num_steps = FLAGS.num_steps // FLAGS.num_envs if \
     FLAGS.use_envpool else FLAGS.num_steps

  environment = helpers.make_environment(
    task,
    use_envpool=use_envpool,
    num_envs=num_envs,
  )
  environment_spec = specs.make_environment_spec(environment)
  network_factory = (
    lambda spec: sac.make_networks(spec, hidden_layer_sizes=(256, 256, 256))
  )

  # Construct the agent.
  config = sac.SACConfig(
    learning_rate=3e-4,
    n_step=2,
    target_entropy=sac.target_entropy_from_env_spec(environment_spec)
  )
  sac_builder = BuilderWrapper(config, num_envs=num_envs)
  # One batch dimension: [batch_size, ...]
  batch_dims = (0,)
  sac_builder = normalization.NormalizationBuilder(
    sac_builder,
    environment_spec,
    is_sequence_based=False,
    batch_dims=batch_dims
  )

  return experiments.Config(
    builder=sac_builder,
    environment_factory=lambda seed: helpers.make_environment(
      task,
      use_envpool,
      num_envs,
    ),
    network_factory=network_factory,
    policy_network_factory=sac.apply_policy_and_sample,
    eval_policy_network_factory=[],
    seed=FLAGS.seed,
    max_number_of_steps=num_steps
  )


def main(_):
  config = build_experiment_config()
  if FLAGS.use_wb:
    run_name = f"acme_sac__{FLAGS.env_name}"
    if FLAGS.use_envpool:
      run_name += f"__envpool-{FLAGS.num_envs}"
    if FLAGS.run_distributed:
      run_name += f"__dist-{FLAGS.num_actors}"
    run_name += f"__{FLAGS.seed}__{int(time.time())}"
    config.logger_factory = partial(
      helpers.make_logger, run_name=run_name, wb_entity=FLAGS.wb_entity
    )

  if FLAGS.run_distributed:
    program = experiments.make_distributed_experiment(
      experiment=config, num_actors=FLAGS.num_actors
    )
    lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))

  experiments.run_experiment(
    experiment=config, eval_every=FLAGS.eval_every, num_eval_episodes=10
  )


if __name__ == "__main__":
  app.run(main)
