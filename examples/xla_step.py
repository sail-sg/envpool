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
"""JAX jitted EnvPool.

See https://envpool.readthedocs.io/en/latest/content/xla_interface.html
"""

import jax.numpy as jnp
from jax import jit, lax

import envpool


def policy(states: jnp.ndarray) -> jnp.ndarray:
  return jnp.zeros(states.shape[0], dtype=jnp.int32)


def gym_sync_step() -> None:
  num_envs = 4
  env = envpool.make_gym("Pong-v5", num_envs=num_envs)

  handle, recv, send, step = env.xla()

  def actor_step(iter, loop_var):
    handle0, states = loop_var
    action = policy(states)
    handle1, (new_states, rew, done, info) = step(handle0, action)
    return (handle1, new_states)

  @jit
  def run_actor_loop(num_steps, init_var):
    return lax.fori_loop(0, num_steps, actor_step, init_var)

  states = env.reset()
  run_actor_loop(100, (handle, states))


def dm_sync_step() -> None:
  num_envs = 4
  env = envpool.make_dm("Pong-v5", num_envs=num_envs)

  handle, recv, send, step = env.xla()

  def actor_step(iter, loop_var):
    handle0, states = loop_var
    action = policy(states.observation.obs)
    handle1, new_states = step(handle0, action)
    return (handle1, new_states)

  @jit
  def run_actor_loop(num_steps, init_var):
    return lax.fori_loop(0, num_steps, actor_step, init_var)

  states = env.reset()
  run_actor_loop(100, (handle, states))


def async_step() -> None:
  num_envs = 8
  batch_size = 4

  # Create an envpool that each step only 4 of 8 result will be out,
  # and left other "slow step" envs execute at background.
  env = envpool.make_dm("Pong-v5", num_envs=num_envs, batch_size=batch_size)

  handle, recv, send, step = env.xla()

  def actor_step(iter, loop_var):
    handle0, states = loop_var
    action = policy(states.observation.obs)
    handle1 = send(handle0, action, states.observation.env_id)
    handle1, new_states = recv(handle0)
    return (handle1, new_states)

  @jit
  def run_actor_loop(num_steps, init_var):
    return lax.fori_loop(0, num_steps, actor_step, init_var)

  env.async_reset()
  handle, states = recv(handle)
  run_actor_loop(100, (handle, states))


if __name__ == "__main__":
  gym_sync_step()
  dm_sync_step()
  async_step()
