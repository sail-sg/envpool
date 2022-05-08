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
from typing import no_type_check

import cv2
import numpy as np
from absl import logging
from absl.testing import absltest

from envpool.atari import AtariDMEnvPool, AtariEnvSpec, AtariGymEnvPool
from envpool.atari.atari_envpool import _AtariEnvPool, _AtariEnvSpec


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

  def test_align(self) -> None:
    """Make sure gym's envpool and dm_env's envpool generate the same data."""
    num_envs = 4
    config = AtariEnvSpec.gen_config(task="space_invaders", num_envs=num_envs)
    spec = AtariEnvSpec(config)
    env0 = AtariGymEnvPool(spec)
    env1 = AtariDMEnvPool(spec)
    obs0 = env0.reset()
    obs1 = env1.reset().observation.obs  # type: ignore
    np.testing.assert_allclose(obs0, obs1)
    for _ in range(1000):
      action = np.random.randint(6, size=num_envs)
      obs0 = env0.step(action)[0]
      obs1 = env1.step(action).observation.obs  # type: ignore
      np.testing.assert_allclose(obs0, obs1)
      # cv2.imwrite(f"/tmp/log/align{i}.png", obs0[0, 1:].transpose(1, 2, 0))

  def test_partial_step(self) -> None:
    num_envs = 5
    max_episode_steps = 10
    config = AtariEnvSpec.gen_config(
      task="defender", num_envs=num_envs, max_episode_steps=max_episode_steps
    )
    spec = AtariEnvSpec(config)
    env = AtariGymEnvPool(spec)
    for _ in range(3):
      print(env)
      env.reset()
      partial_ids = [np.arange(num_envs)[::2], np.arange(num_envs)[1::2]]
      env.step(np.zeros(len(partial_ids[1]), dtype=int), env_id=partial_ids[1])
      for _ in range(max_episode_steps - 2):
        info = env.step(
          np.zeros(num_envs, dtype=int), env_id=np.arange(num_envs)
        )[-1]
        assert np.all(~info["TimeLimit.truncated"])
      info = env.step(
        np.zeros(num_envs, dtype=int), env_id=np.arange(num_envs)
      )[-1]
      env_id = np.array(info["env_id"])
      done_id = np.array(sorted(env_id[info["TimeLimit.truncated"]]))
      assert np.all(done_id == partial_ids[1])
      info = env.step(
        np.zeros(len(partial_ids[0]), dtype=int),
        env_id=partial_ids[0],
      )[-1]
      assert np.all(info["TimeLimit.truncated"])

  def test_xla_step(self) -> None:
    num_envs = 10
    config = AtariEnvSpec.gen_config(
      task="pong",
      num_envs=num_envs,
      batch_size=5,
      num_threads=2,
      thread_affinity_offset=0,
    )
    spec = AtariEnvSpec(config)
    env = AtariGymEnvPool(spec)
    handle, recv, send, step = env.xla()
    env.reset()
    handle, states = recv(handle)
    action = np.ones(5, dtype=np.int32)
    print(states)
    handle = send(handle, action)

  @no_type_check
  def test_no_gray_scale(self) -> None:
    ref_shape = (12, 84, 84)
    raw_shape = (12, 210, 160)
    config = AtariEnvSpec.gen_config(task="breakout", gray_scale=False)
    spec = AtariEnvSpec(config)
    env = AtariGymEnvPool(spec)
    self.assertTrue(env.observation_space.shape, ref_shape)
    obs = env.reset()
    self.assertTrue(obs.shape, ref_shape)
    config = AtariEnvSpec.gen_config(
      task="breakout", gray_scale=False, img_height=210, img_width=160
    )
    spec = AtariEnvSpec(config)
    env = AtariGymEnvPool(spec)
    self.assertTrue(env.observation_space.shape, raw_shape)
    obs1 = env.reset()
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
    config = AtariEnvSpec.gen_config(
      task="pong",
      num_envs=num_envs,
      batch_size=batch,
      num_threads=num_threads,
      thread_affinity_offset=0,
    )
    spec = AtariEnvSpec(config)
    env = AtariGymEnvPool(spec)
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
