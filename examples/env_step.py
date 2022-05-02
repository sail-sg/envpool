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

import numpy as np

import envpool


def gym_sync_step() -> None:
  num_envs = 4
  env = envpool.make_gym("Pong-v5", num_envs=num_envs)
  action_num = env.action_space.n
  obs = env.reset()  # reset all envs
  assert obs.shape == (num_envs, 4, 84, 84)
  for _ in range(1000):
    # autoreset is automatically enabled in envpool
    action = np.random.randint(action_num, size=num_envs)
    obs, rew, done, info = env.step(action)
  # Of course, you can specify env_id to step corresponding envs
  obs = env.reset(np.array([1, 3]))  # reset env #1 and #3
  assert obs.shape == (2, 4, 84, 84)
  partial_action = np.array([0, 0, 2])
  env_id = np.array([3, 2, 0])
  obs, rew, done, info = env.step(partial_action, env_id)
  np.testing.assert_allclose(info["env_id"], env_id)
  assert obs.shape == (3, 4, 84, 84)


def dm_sync_step() -> None:
  num_envs = 4
  env = envpool.make_dm("Pong-v5", num_envs=num_envs)
  action_num = env.action_spec().num_values
  ts = env.reset()
  # ts.observation is a **NamedTuple** instead of np.ndarray
  # because we need to store other valuable information in this field
  assert ts.observation.obs.shape, (num_envs, 4, 84, 84)
  for _ in range(1000):
    # autoreset is automatically enabled in envpool
    action = np.random.randint(action_num, size=num_envs)
    ts = env.step(action)
  # Of course, you can specify env_id to step corresponding envs
  ts = env.reset(np.array([1, 3]))  # reset env #1 and #3
  assert ts.observation.obs.shape == (2, 4, 84, 84)
  partial_action = np.array([0, 0, 2])
  env_id = np.array([3, 2, 0])
  ts = env.step(partial_action, env_id)
  np.testing.assert_allclose(ts.observation.env_id, env_id)
  assert ts.observation.obs.shape == (3, 4, 84, 84)


def async_step() -> None:
  num_envs = 8
  batch_size = 4

  # Create an envpool that each step only 4 of 8 result will be out,
  # and left other "slow step" envs execute at background.
  env = envpool.make_dm("Pong-v5", num_envs=num_envs, batch_size=batch_size)
  action_num = env.action_spec().num_values
  ts = env.reset()
  for _ in range(1000):
    env_id = ts.observation.env_id
    assert len(env_id) == batch_size
    # generate action with len(action) == len(env_id)
    action = np.random.randint(action_num, size=batch_size)
    ts = env.step(action, env_id)

  # Same as gym
  env = envpool.make_gym(
    "Pong-v5",
    num_envs=num_envs,
    batch_size=batch_size,
    gym_reset_return_info=True,
  )
  # If you want gym's reset() API return env_id,
  # just set gym_reset_return_info=True
  obs, info = env.reset()
  assert obs.shape == (batch_size, 4, 84, 84)
  env_id = info["env_id"]
  for _ in range(1000):
    action = np.random.randint(action_num, size=batch_size)
    obs, rew, done, info = env.step(action, env_id)
    env_id = info["env_id"]
    assert len(env_id) == batch_size
    assert obs.shape == (batch_size, 4, 84, 84)

  # We can also use a low-level API (send and recv)
  env = envpool.make_gym("Pong-v5", num_envs=num_envs, batch_size=batch_size)
  env.async_reset()  # no return, just send `reset` signal to all envs
  for _ in range(1000):
    obs, rew, done, info = env.recv()
    env_id = info["env_id"]
    assert len(env_id) == batch_size
    assert obs.shape == (batch_size, 4, 84, 84)
    action = np.random.randint(action_num, size=batch_size)
    env.send(action, env_id)


if __name__ == "__main__":
  gym_sync_step()
  dm_sync_step()
  async_step()
