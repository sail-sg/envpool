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
"""Example running PPO on Mujoco tasks using EnvPool."""

import argparse
import logging
import os
from dataclasses import asdict
from functools import partial

import helpers
import jax
from acme.agents.jax import ppo
from acme.jax import experiments


def parse_args():
  # fmt: off
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--exp-name",
    type=str,
    default=os.path.basename(__file__).rstrip(".py"),
    help="Name of this experiment.",
  )
  parser.add_argument(
    "--env-name",
    type=str,
    default="HalfCheetah-v3",
    help="What environment to run.",
  )
  parser.add_argument(
    "--use-envpool",
    type=bool,
    default=False,
    nargs="?",
    const=True,
    help="Whether to use EnvPool.",
  )
  parser.add_argument(
    "--use-vec-env",
    type=bool,
    default=False,
    nargs="?",
    const=True,
    help="Whether to use SB3 VecEnv.",
  )
  parser.add_argument(
    "--num-envs",
    type=int,
    default=8,
    help="Number of environments (EnvPool/DummyVecEnv).",
  )
  parser.add_argument("--seed", type=int, default=0, help="Random seed.")
  parser.add_argument(
    "--num-steps",
    type=int,
    default=1_000_000,
    help="Number of env steps to run.",
  )
  parser.add_argument(
    "--use-wb",
    type=bool,
    default=False,
    nargs="?",
    const=True,
    help="Whether to use WandB.",
  )
  parser.add_argument(
    "--wb-project", type=str, default="acme", help="W&B project name."
  )
  parser.add_argument(
    "--wb-entity", type=str, default=None, help="W&B entity name."
  )
  args = parser.parse_args()
  return args


def build_experiment_config(FLAGS):
  task = FLAGS.env_name

  use_envpool = FLAGS.use_envpool
  use_vec_env = not use_envpool and FLAGS.use_vec_env
  use_batch_env = use_envpool or use_vec_env

  num_envs = FLAGS.num_envs if use_batch_env else -1
  num_steps = FLAGS.num_steps // FLAGS.num_envs if \
    use_batch_env else FLAGS.num_steps

  config = ppo.PPOConfig()
  ppo_builder = helpers.PPOBuilder(config, num_envs)

  config = asdict(config)
  config["use_batch_env"] = use_batch_env

  layer_sizes = (256, 256, 256)

  return experiments.Config(
    builder=ppo_builder,
    environment_factory=lambda _: helpers.make_mujoco_environment(
      task,
      use_envpool=use_envpool,
      use_vec_env=use_vec_env,
      num_envs=num_envs,
    ),
    network_factory=lambda spec: ppo.make_networks(spec, layer_sizes),
    policy_network_factory=ppo.make_inference_fn,
    evaluator_factories=[],
    seed=0,
    max_number_of_steps=num_steps - 1,
  ), config


def main():
  logging.info(f"Jax Devices: {jax.devices()}")
  FLAGS = parse_args()
  experiment, config = build_experiment_config(FLAGS)
  if FLAGS.use_wb:
    run_name = f"acme_ppo__{FLAGS.env_name}"
    if FLAGS.use_envpool:
      run_name += f"__envpool-{FLAGS.num_envs}"
    elif FLAGS.use_vec_env:
      run_name += f"__vec_env-{FLAGS.num_envs}"
    run_name += f"__seed-{FLAGS.seed}"
    config.update(vars(FLAGS))
    experiment.logger_factory = partial(
      helpers.make_logger,
      config=config,
      run_name=run_name,
      wb_entity=FLAGS.wb_entity,
    )

  experiments.run_experiment(
    experiment=experiment,
    eval_every=experiment.max_number_of_steps,
  )


if __name__ == "__main__":
  main()
