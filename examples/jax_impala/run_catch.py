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

import functools
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
from absl import app
from acme import wrappers

import envpool
from envpool.python.protocol import EnvPool

USE_ENVPOOL = True

ACTION_REPEAT = 1
BATCH_SIZE = 16
DISCOUNT_FACTOR = 0.99
MAX_ENV_FRAMES = 10_000_000
NUM_ACTORS = 8
UNROLL_LENGTH = 20

FRAMES_PER_ITER = ACTION_REPEAT * BATCH_SIZE * UNROLL_LENGTH


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


def make_catch_envpool(num_envs: int = 2,) -> EnvPool:
  env = envpool.make(
    "Catch-v0",
    env_type="dm",
    num_envs=num_envs,
  )
  return env


def main(_):
  # A thunk that builds a new environment.
  # Substitute your environment here!
  build_env = make_atari_environment
  if USE_ENVPOOL:
    build_env = functools.partial(make_catch_envpool, num_envs=BATCH_SIZE)

  # Construct the agent. We need a sample environment for its spec.
  env_for_spec = build_env()
  num_actions = env_for_spec.action_spec().num_values
  obs_spec = env_for_spec.observation_spec()
  if USE_ENVPOOL:
    obs_spec = obs_spec.obs
  agent = agent_lib.Agent(
    num_actions, obs_spec,
    functools.partial(haiku_nets.CatchNet)
  )

  # Construct the optimizer.
  max_updates = MAX_ENV_FRAMES / FRAMES_PER_ITER
  opt = optax.rmsprop(5e-3, decay=0.99, eps=1e-7)

  # Construct the learner.
  learner_builder = learner_lib.Learner
  if USE_ENVPOOL:
    learner_builder = functools.partial(
      learner_lib.BatchLearner, num_envs=BATCH_SIZE
    )
  learner = learner_builder(
    agent=agent,
    rng_key=jax.random.PRNGKey(428),
    opt=opt,
    batch_size=BATCH_SIZE,
    discount_factor=DISCOUNT_FACTOR,
    frames_per_iter=FRAMES_PER_ITER,
    max_abs_reward=1.,
    logger=util.AbslLogger(),  # Provide your own logger here.
  )

  # Construct the actors on different threads.
  # stop_signal in a list so the reference is shared.
  actor_threads = []
  stop_signal = [False]
  actor_builder = actor_lib.Actor
  if USE_ENVPOOL:
    actor_builder = functools.partial(
      actor_lib.BatchActor, num_envs=BATCH_SIZE
    )
    num_actors = 1
  else:
    num_actors = NUM_ACTORS
  for i in range(num_actors):
    actor = actor_builder(
      agent=agent,
      env=build_env(),
      unroll_length=UNROLL_LENGTH,
      learner=learner,
      rng_seed=i,
      logger=util.AbslLogger(),  # Provide your own logger here.
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


if __name__ == '__main__':
  app.run(main)
