# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Single-process IMPALA wiring."""

import argparse
import functools
import os
import threading
from typing import List

import actor as actor_lib
import agent as agent_lib
import dm_env
import gym
import haiku_nets
import jax
import learner as learner_lib
import optax
import util
from acme import wrappers

import envpool
from envpool.python.protocol import EnvPool

NUM_ACTORS = 8


def run_actor(actor: actor_lib.Actor, stop_signal: List[bool]):
  """Runs an actor to produce num_trajectories trajectories."""
  while not stop_signal[0]:
    frame_count, params = actor.pull_params()
    actor.unroll_and_push(frame_count, params)


def make_atari_environment(
  level: str = 'Pong',
  sticky_actions: bool = True,
  zero_discount_on_life_loss: bool = False
) -> dm_env.Environment:
  """Loads the Atari environment."""
  version = 'v0' if sticky_actions else 'v4'
  level_name = f'{level}NoFrameskip-{version}'
  env = gym.make(level_name, full_action_space=True)

  wrapper_list = [
    wrappers.GymAtariAdapter,
    functools.partial(
      wrappers.AtariWrapper,
      to_float=True,
      max_episode_len=108_000,
      zero_discount_on_life_loss=zero_discount_on_life_loss,
    ),
  ]

  wrapper_list.append(wrappers.SinglePrecisionWrapper)

  return wrappers.wrap_all(env, wrapper_list)


def make_atari_envpool(
  task: str = "Pong-v5",
  num_envs: int = 2,
) -> EnvPool:
  env = envpool.make(
    task,
    env_type="dm",
    num_envs=num_envs,
    episodic_life=True,
    reward_clip=True
  )
  return env


def main(FLAGS):
  cfg_dict = vars(FLAGS)
  cfg_dict.update(
    {
      "action_repeat": 1,
      "batch_size": FLAGS.batch_size,
      "discount_factor": FLAGS.discount_factor,
      "unroll_length": FLAGS.unroll_length,
    }
  )
  run_name = ""
  if FLAGS.use_wb:
    run_name += f"jax_impala__{FLAGS.env_name}"
    if FLAGS.use_envpool:
      run_name += f"__envpool-{FLAGS.num_envs}"
    run_name += f"__seed-{FLAGS.seed}"
  # Environment builder.
  build_env = make_atari_environment
  if FLAGS.use_envpool:
    build_env = functools.partial(make_atari_envpool, num_envs=FLAGS.num_envs)

  # Construct the agent. We need a sample environment for its spec.
  env_for_spec = build_env()
  num_actions = env_for_spec.action_spec().num_values
  obs_spec = env_for_spec.observation_spec()
  if FLAGS.use_envpool:
    obs_spec = obs_spec.obs
  agent = agent_lib.Agent(
    num_actions, obs_spec,
    functools.partial(haiku_nets.AtariNet, use_resnet=True, use_lstm=False)
  )

  # Construct the optimizer.
  frames_per_iter = FLAGS.batch_size * FLAGS.unroll_length
  max_updates = FLAGS.num_steps / frames_per_iter
  opt = optax.rmsprop(6e-4, decay=0.99, eps=1e-7)

  # Construct the learner.
  learner_builder = learner_lib.Learner
  if FLAGS.use_envpool:
    learner_builder = functools.partial(
      learner_lib.BatchLearner, num_envs=FLAGS.num_envs
    )
  learner = learner_builder(
    agent=agent,
    rng_key=jax.random.PRNGKey(FLAGS.seed),
    opt=opt,
    batch_size=FLAGS.batch_size,
    discount_factor=FLAGS.discount_factor,
    frames_per_iter=frames_per_iter,
    max_abs_reward=1.,
    logger=util.AbslLogger(),  # Provide your own logger here.
  )

  # Construct the actors on different threads.
  # stop_signal in a list so the reference is shared.
  actor_threads = []
  stop_signal = [False]
  actor_builder = actor_lib.Actor
  if FLAGS.use_envpool:
    actor_builder = functools.partial(
      actor_lib.BatchActor, num_envs=FLAGS.num_envs
    )
    num_actors = 1
  else:
    num_actors = NUM_ACTORS
  for i in range(num_actors):
    actor = actor_builder(
      agent=agent,
      env=build_env(),
      unroll_length=FLAGS.unroll_length,
      learner=learner,
      rng_seed=i,
      logger=util.WBLogger(
        cfg_dict["wb_project"], cfg_dict["wb_entity"], run_name, cfg_dict,
        cfg_dict["use_wb"]
      ),  # Provide your own logger here.
    )
    args = (actor, stop_signal)
    actor_threads.append(threading.Thread(target=run_actor, args=args))

  # Start the actors and learner.
  for t in actor_threads:
    t.start()
  learner.run(int(max_updates))

  # Stop.
  stop_signal[0] = True
  for t in actor_threads:
    t.join()


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
    default=32,
    help="Number of environments (EnvPool/DummyVecEnv)."
  )
  parser.add_argument(
    "--batch-size", type=int, default=32, help="Batch size of learner."
  )
  parser.add_argument(
    "--unroll-length",
    type=int,
    default=20,
    help="Unroll length of trajectory."
  )
  parser.add_argument(
    "--discount-factor", type=float, default=0.99, help="Discount factor."
  )
  parser.add_argument("--seed", type=int, default=0, help="Random seed.")
  parser.add_argument(
    "--num-steps",
    type=int,
    default=50_000_000,
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


if __name__ == '__main__':
  main(parse_args())
