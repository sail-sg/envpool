# Copyright 2023 Garena Online Private Limited
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

import jax

import envpool

env = envpool.make(
  "Pong-v5",
  env_type="dm",
  num_envs=2,
)
handle, recv, send, _ = env.xla()


def actor_step(iter, loop_var):
  handle0, states = loop_var
  action = 0
  handle1 = send(handle0, action, states.observation.env_id)
  handle1, new_states = recv(handle0)
  return handle1, new_states


@jax.jit
def run_actor_loop(num_steps, init_var):
  return jax.lax.fori_loop(0, num_steps, actor_step, init_var)


env.async_reset()
handle, states = recv(handle)
run_actor_loop(100, (handle, states))
