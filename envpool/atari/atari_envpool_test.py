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
"""Unit test for atari envpool and speed benchmark."""

import os
import time

import cv2
import jax.numpy as jnp
import numpy as np
from absl import logging
from absl.testing import absltest
from jax import jit, lax

import envpool.atari.registration  # noqa: F401
from envpool.atari.atari_envpool import _AtariEnvPool, _AtariEnvSpec
from envpool.registration import make_dm, make_gym, make_gymnasium


class _AtariEnvPoolTest(absltest.TestCase):

  def test_raw_envpool(self) -> None:
    conf = dict(
      zip(_AtariEnvSpec._config_keys, _AtariEnvSpec._default_config_values)
    )
    conf["task"] = b"pong"
    conf["num_envs"] = num_envs = 3
    conf["batch_size"] = batch = 3
    conf["num_threads"] = 3  # os.cpu_count()
    # conf["episodic_life"] = True
    # conf["zero_discount_on_life_loss"] = True
    env_spec = _AtariEnvSpec(tuple(conf.values()))
    env = _AtariEnvPool(env_spec)
    env._reset(np.arange(num_envs, dtype=np.int32))
    state_keys = env._state_keys
    total = 2000
    actions = np.random.randint(6, size=(total, batch))
    t = time.time()
    for i in range(total):
      state = dict(zip(state_keys, env._recv()))
      # obs = state["obs"]
      # cv2.imwrite(f"/tmp/log/raw{i}.png", obs[0, 1:].transpose(1, 2, 0))
      action = {
        "env_id": state["info:env_id"],
        "players.env_id": state["info:players.env_id"],
        "action": actions[i],
      }
      env._send(tuple(action.values()))
    duration = time.time() - t
    fps = total * batch / duration * 4
    logging.info(f"Raw envpool FPS = {fps:.6f}")

  def test_full_action_space(self) -> None:
    env = make_gym("Pong-v5", full_action_space=True)
    self.assertEqual(env.action_space.n, 18)
    env = make_gym("Breakout-v5", full_action_space=True)
    self.assertEqual(env.action_space.n, 18)

  def test_align(self) -> None:
    """Make sure gym's envpool and dm_env's envpool generate the same data."""
    num_envs = 4
    env0 = make_gym("SpaceInvaders-v5", num_envs=num_envs)
    env1 = make_dm("SpaceInvaders-v5", num_envs=num_envs)
    env2 = make_gymnasium("SpaceInvaders-v5", num_envs=num_envs)
    obs0, _ = env0.reset()
    obs1 = env1.reset().observation.obs
    obs2, _ = env2.reset()
    np.testing.assert_allclose(obs0, obs1)
    np.testing.assert_allclose(obs1, obs2)
    for _ in range(1000):
      action = np.random.randint(6, size=num_envs)
      obs0 = env0.step(action)[0]
      obs1 = env1.step(action).observation.obs
      obs2 = env2.step(action)[0]
      np.testing.assert_allclose(obs0, obs1)
      np.testing.assert_allclose(obs1, obs2)
      # cv2.imwrite(f"/tmp/log/align{i}.png", obs0[0, 1:].transpose(1, 2, 0))

  def test_reset_life(self) -> None:
    """Issue 171."""
    for env_id in [
      "atlantis", "backgammon", "breakout", "pong", "wizard_of_wor"
    ]:
      np.random.seed(0)
      task_id = "".join([g.capitalize() for g in env_id.split("_")]) + "-v5"
      env = make_gym(task_id, episodic_life=True)
      action_num = env.action_space.n
      env.reset()
      info = env.step(np.array([0]))[-1]
      if info["lives"].sum() == 0:
        # no life in this game
        continue
      for _ in range(10000):
        _, _, terminated, truncated, info = env.step(
          np.random.randint(0, action_num, 1)
        )
        done = np.logical_or(terminated, truncated)
        if info["lives"][0] == 0:
          break
        else:
          self.assertFalse(info["terminated"][0])
      if info["lives"][0] > 0:
        # step too long
        continue
      # for normal atari (e.g., breakout)
      # take an additional step after all lives are exhausted
      _, _, next_terminated, next_truncated, next_info = env.step(
        np.random.randint(0, action_num, 1)
      )
      if done[0] and next_info["lives"][0] > 0:
        self.assertTrue(info["terminated"][0])
        continue
      self.assertFalse(done[0])
      self.assertFalse(info["terminated"][0])
      while not done[0]:
        self.assertFalse(info["terminated"][0])
        _, _, terminated, truncated, info = env.step(
          np.random.randint(0, action_num, 1)
        )
        done = np.logical_or(terminated, truncated)
      _, _, next_terminated, next_truncated, next_info = env.step(
        np.random.randint(0, action_num, 1)
      )
      self.assertTrue(next_info["lives"][0] > 0)
      self.assertTrue(info["terminated"][0])

  def test_partial_step(self) -> None:
    num_envs = 5
    max_episode_steps = 10
    env = make_gym(
      "Defender-v5", num_envs=num_envs, max_episode_steps=max_episode_steps
    )
    for _ in range(3):
      print(env)
      env.reset()
      partial_ids = [np.arange(num_envs)[::2], np.arange(num_envs)[1::2]]
      env.step(np.zeros(len(partial_ids[1]), dtype=int), env_id=partial_ids[1])
      for _ in range(max_episode_steps - 2):
        _, _, _, truncated, info = env.step(
          np.zeros(num_envs, dtype=int), env_id=np.arange(num_envs)
        )
        assert np.all(~truncated)
      _, _, _, truncated, info = env.step(
        np.zeros(num_envs, dtype=int), env_id=np.arange(num_envs)
      )
      env_id = np.array(info["env_id"])
      done_id = np.array(sorted(env_id[truncated]))
      assert np.all(done_id == partial_ids[1])
      _, _, _, truncated, info = env.step(
        np.zeros(len(partial_ids[0]), dtype=int),
        env_id=partial_ids[0],
      )
      assert np.all(truncated)

  def test_xla_api(self) -> None:
    env = make_gym(
      "Pong-v5",
      num_envs=10,
      batch_size=5,
      num_threads=2,
      thread_affinity_offset=0,
    )
    handle, recv, send, step = env.xla()
    env.async_reset()
    handle, states = recv(handle)
    info = states[-1]
    action = np.ones(5, dtype=np.int32)
    handle = send(handle, action, env_id=info["env_id"])

    def actor_step(iter: int, handle: jnp.ndarray) -> jnp.ndarray:
      handle, states = recv(handle)
      info = states[-1]
      action = jnp.ones(5, dtype=jnp.int32)
      handle = send(handle, action, env_id=info["env_id"])
      return handle

    @jit
    def loop(num_steps: int = 100) -> jnp.ndarray:
      return lax.fori_loop(0, num_steps, actor_step, handle)

    loop(100)

  def test_xla_correctness(self) -> None:
    env1 = make_gym(
      "Pong-v5",
      num_envs=10,
      batch_size=10,
      num_threads=2,
      thread_affinity_offset=0,
    )
    env2 = make_gym(
      "Pong-v5",
      num_envs=10,
      batch_size=10,
      num_threads=2,
      thread_affinity_offset=0,
    )
    handle, recv, send, step = env1.xla()
    env1.async_reset()
    env2.async_reset()

    action = np.ones(10, dtype=np.int32)
    for _ in range(100):
      handle, states1 = recv(handle)
      handle = send(handle, action)
      states2 = env2.recv()
      env2.send(action)
      np.testing.assert_allclose(states1[0], states2[0])

  def test_no_gray_scale(self) -> None:
    ref_shape = (12, 84, 84)
    raw_shape = (12, 210, 160)
    env = make_gym("Breakout-v5", gray_scale=False)
    self.assertTrue(env.observation_space.shape, ref_shape)
    obs, _ = env.reset()
    self.assertTrue(obs.shape, ref_shape)
    env = make_gym(
      "Breakout-v5", gray_scale=False, img_height=210, img_width=160
    )
    self.assertTrue(env.observation_space.shape, raw_shape)
    obs1, _ = env.reset()
    self.assertTrue(obs1.shape, raw_shape)
    for i in range(0, 12, 3):
      obs_ = cv2.resize(
        obs1[0, i:i + 3].transpose(1, 2, 0), (84, 84),
        interpolation=cv2.INTER_AREA
      )
      np.testing.assert_allclose(obs_, obs[0, i:i + 3].transpose(1, 2, 0))

  def test_benchmark(self) -> None:
    if os.cpu_count() == 256:
      num_envs = 645
      batch = 248
      num_threads = 248
      total = 50000
    else:
      num_envs = 8
      batch = 3
      num_threads = 3
      total = 1000
    env = make_gym(
      "Pong-v5",
      num_envs=num_envs,
      batch_size=batch,
      num_threads=num_threads,
      thread_affinity_offset=0,
    )
    env.async_reset()
    action = np.ones(batch, dtype=np.int32)
    t = time.time()
    for _ in range(total):
      info = env.recv()[-1]
      env.send(action, info["env_id"])
    duration = time.time() - t
    fps = total * batch / duration * 4
    logging.info(f"Python envpool FPS = {fps:.6f}")


if __name__ == "__main__":
  absltest.main()
