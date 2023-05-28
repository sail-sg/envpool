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

from typing import no_type_check

import dm_env
import gym
import numpy as np
from absl import logging
from absl.testing import absltest
from packaging import version

import envpool.atari.registration  # noqa: F401
from envpool.atari import AtariEnvSpec
from envpool.registration import make_dm, make_gym, make_spec


class _SpecTest(absltest.TestCase):

  @no_type_check
  def test_spec(self) -> None:
    action_nums = {"pong": 6, "breakout": 4}
    for task in ["pong", "breakout"]:
      action_num = action_nums[task]
      spec = make_spec(task.capitalize() + "-v5")
      logging.info(spec)
      self.assertEqual(spec.action_array_spec["action"].maximum + 1, action_num)
      # check dm spec
      dm_obs_spec = spec.observation_spec().obs
      dm_act_spec = spec.action_spec()
      self.assertEqual(len(spec.action_array_spec), 3)
      self.assertIsInstance(dm_obs_spec, dm_env.specs.BoundedArray)
      self.assertEqual(dm_obs_spec.dtype, np.uint8)
      self.assertEqual(dm_obs_spec.maximum, 255)
      self.assertIsInstance(dm_act_spec, dm_env.specs.DiscreteArray)
      self.assertEqual(dm_act_spec.num_values, action_num)
      # check gym space
      gym_obs_space: gym.spaces.Box = spec.observation_space
      gym_act_space: gym.spaces.Discrete = spec.action_space
      self.assertEqual(len(spec.action_array_spec), 3)
      self.assertIsInstance(gym_obs_space, gym.spaces.Box)
      self.assertEqual(gym_obs_space.dtype, np.uint8)
      np.testing.assert_allclose(gym_obs_space.high, 255)
      self.assertIsInstance(gym_act_space, gym.spaces.Discrete)
      self.assertEqual(gym_act_space.n, action_num)

  def test_seed_warning(self) -> None:
    num_envs = 4
    env = make_dm("Pong-v5", num_envs=num_envs)
    with self.assertWarns(UserWarning):
      env.seed(1)
    env = make_gym("Pong-v5", num_envs=num_envs)
    with self.assertWarns(UserWarning):
      env.seed()

  def test_invalid_batch_size(self) -> None:
    num_envs = 4
    batch_size = 5
    config = AtariEnvSpec.gen_config(
      task="pong", num_envs=num_envs, batch_size=batch_size
    )
    self.assertRaises(ValueError, AtariEnvSpec, config)

  def test_metadata(self) -> None:
    num_envs = 4
    env = make_gym("Pong-v5", num_envs=num_envs)
    self.assertEqual(len(env), num_envs)
    self.assertFalse(env.is_async)
    num_envs = 8
    batch_size = 4
    env = make_gym("Pong-v5", num_envs=num_envs, batch_size=batch_size)
    self.assertEqual(len(env), num_envs)
    self.assertTrue(env.is_async)
    self.assertIsNone(env.spec.reward_threshold)


class _DMSyncTest(absltest.TestCase):

  @no_type_check
  def test_spec(self) -> None:
    action_nums = {"pong": 6, "breakout": 4}
    for task in ["pong", "breakout"]:
      action_num = action_nums[task]
      env = make_dm(task.capitalize() + "-v5")
      self.assertIsInstance(env, dm_env.Environment)
      logging.info(env)
      # check dm spec
      dm_obs_spec = env.observation_spec().obs
      dm_act_spec = env.action_spec()
      self.assertIsInstance(dm_obs_spec, dm_env.specs.BoundedArray)
      self.assertEqual(dm_obs_spec.dtype, np.uint8)
      self.assertEqual(dm_obs_spec.maximum, 255)
      self.assertIsInstance(dm_act_spec, dm_env.specs.DiscreteArray)
      self.assertEqual(dm_act_spec.num_values, action_num)

  def test_lowlevel_step(self) -> None:
    num_envs = 4
    env = make_dm("Pong-v5", num_envs=num_envs)
    logging.info(env)
    env.async_reset()
    ts: dm_env.TimeStep = env.recv()
    # check ts structure
    self.assertTrue(np.all(ts.first()))
    np.testing.assert_allclose(ts.step_type.shape, (num_envs,))
    np.testing.assert_allclose(ts.reward.shape, (num_envs,))
    self.assertEqual(ts.reward.dtype, np.float32)
    np.testing.assert_allclose(ts.discount.shape, (num_envs,))
    self.assertEqual(ts.discount.dtype, np.float32)
    np.testing.assert_allclose(ts.observation.obs.shape, (num_envs, 4, 84, 84))
    self.assertEqual(ts.observation.obs.dtype, np.uint8)
    np.testing.assert_allclose(ts.observation.lives.shape, (num_envs,))
    self.assertEqual(ts.observation.lives.dtype, np.int32)
    np.testing.assert_allclose(ts.observation.env_id, np.arange(num_envs))
    self.assertEqual(ts.observation.env_id.dtype, np.int32)
    np.testing.assert_allclose(ts.observation.players.env_id.shape, (num_envs,))
    self.assertEqual(ts.observation.players.env_id.dtype, np.int32)
    action = {
      "env_id": np.arange(num_envs),
      "players.env_id": np.arange(num_envs),
      "action": np.ones(num_envs, int)
    }
    # because in c++ side we define action is int32 instead of int64
    self.assertRaises(RuntimeError, env.send, action)
    action = {
      "env_id": np.arange(num_envs, dtype=np.int32),
      "players.env_id": np.arange(num_envs, dtype=np.int32),
      "action": np.ones(num_envs, np.int32)
    }
    env.send(action)
    ts1: dm_env.TimeStep = env.recv()
    self.assertTrue(np.all(ts1.mid()))
    action = np.ones(num_envs)
    env.send(action)
    ts2: dm_env.TimeStep = env.recv()
    self.assertTrue(np.all(ts2.mid()))
    while np.all(ts2.mid()):
      env.send(np.random.randint(6, size=num_envs))
      ts2 = env.recv()
    env.send(np.random.randint(6, size=num_envs))
    tsp1: dm_env.TimeStep = env.recv()
    index = np.where(ts2.last())
    np.testing.assert_allclose(ts2.discount[index], 0)
    np.testing.assert_allclose(tsp1.step_type[index], dm_env.StepType.FIRST)
    np.testing.assert_allclose(tsp1.discount[index], 1)

  def test_highlevel_step(self) -> None:
    num_envs = 4
    # defender game hangs infinitely in gym.make("Defender-v0")
    env = make_dm("Defender-v5", num_envs=num_envs)
    logging.info(env)
    ts: dm_env.TimeStep = env.reset()
    # check ts structure
    self.assertTrue(np.all(ts.first()))
    np.testing.assert_allclose(ts.step_type.shape, (num_envs,))
    np.testing.assert_allclose(ts.reward.shape, (num_envs,))
    self.assertEqual(ts.reward.dtype, np.float32)
    np.testing.assert_allclose(ts.discount.shape, (num_envs,))
    self.assertEqual(ts.discount.dtype, np.float32)
    np.testing.assert_allclose(ts.observation.obs.shape, (num_envs, 4, 84, 84))
    self.assertEqual(ts.observation.obs.dtype, np.uint8)
    np.testing.assert_allclose(ts.observation.lives.shape, (num_envs,))
    self.assertEqual(ts.observation.lives.dtype, np.int32)
    np.testing.assert_allclose(ts.observation.env_id, np.arange(num_envs))
    self.assertEqual(ts.observation.env_id.dtype, np.int32)
    np.testing.assert_allclose(ts.observation.players.env_id.shape, (num_envs,))
    self.assertEqual(ts.observation.players.env_id.dtype, np.int32)
    action = {
      "env_id": np.arange(num_envs),
      "players.env_id": np.arange(num_envs),
      "action": np.ones(num_envs, int)
    }
    # because in c++ side we define action is int32 instead of int64
    self.assertRaises(RuntimeError, env.step, action)
    action = {
      "env_id": np.arange(num_envs, dtype=np.int32),
      "players.env_id": np.arange(num_envs, dtype=np.int32),
      "action": np.ones(num_envs, np.int32)
    }
    ts1: dm_env.TimeStep = env.step(action)
    self.assertTrue(np.all(ts1.mid()))
    action = np.ones(num_envs)
    ts2: dm_env.TimeStep = env.step(action)
    self.assertTrue(np.all(ts2.mid()))
    while np.all(ts2.mid()):
      ts2 = env.step(np.random.randint(18, size=num_envs))
    tsp1: dm_env.TimeStep = env.step(np.random.randint(18, size=num_envs))
    index = np.where(ts2.last())
    np.testing.assert_allclose(ts2.discount[index], 0)
    np.testing.assert_allclose(tsp1.step_type[index], dm_env.StepType.FIRST)
    np.testing.assert_allclose(tsp1.discount[index], 1)


class _GymSyncTest(absltest.TestCase):

  @no_type_check
  def test_spec(self) -> None:
    action_nums = {"pong": 6, "breakout": 4}
    for task in ["pong", "breakout"]:
      action_num = action_nums[task]
      env = make_gym(task.capitalize() + "-v5")
      self.assertIsInstance(env, gym.Env)
      logging.info(env)
      # check gym space
      gym_obs_space: gym.spaces.Box = env.observation_space
      gym_act_space: gym.spaces.Discrete = env.action_space
      self.assertEqual(len(env.spec.action_array_spec), 3)
      self.assertIsInstance(gym_obs_space, gym.spaces.Box)
      self.assertEqual(gym_obs_space.dtype, np.uint8)
      np.testing.assert_allclose(gym_obs_space.high, 255)
      self.assertIsInstance(gym_act_space, gym.spaces.Discrete)
      self.assertEqual(gym_act_space.n, action_num)
      # Issue 207
      gym_act_space.seed(1)
      action0 = gym_act_space.sample()
      gym_act_space.seed(1)
      action1 = gym_act_space.sample()
      self.assertEqual(action0, action1)
      env.action_space.seed(2)
      action2 = env.action_space.sample()
      env.action_space.seed(2)
      action3 = env.action_space.sample()
      self.assertEqual(action2, action3)

  def test_lowlevel_step(self) -> None:
    num_envs = 4
    env = make_gym("Breakout-v5", num_envs=num_envs)
    self.assertTrue(isinstance(env, gym.Env))
    logging.info(env)
    env.async_reset()
    obs, rew, terminated, truncated, info = env.recv()
    done = np.logical_or(terminated, truncated)
    # check shape
    self.assertIsInstance(obs, np.ndarray)
    self.assertEqual(obs.dtype, np.uint8)
    np.testing.assert_allclose(rew.shape, (num_envs,))
    self.assertEqual(rew.dtype, np.float32)
    np.testing.assert_allclose(done.shape, (num_envs,))
    self.assertEqual(done.dtype, np.bool_)
    self.assertEqual(terminated.dtype, np.bool_)
    self.assertEqual(truncated.dtype, np.bool_)
    self.assertIsInstance(info, dict)
    self.assertEqual(len(info), 6)
    self.assertEqual(info["env_id"].dtype, np.int32)
    self.assertEqual(info["lives"].dtype, np.int32)
    self.assertEqual(info["players"]["env_id"].dtype, np.int32)
    np.testing.assert_allclose(info["env_id"], np.arange(num_envs))
    np.testing.assert_allclose(info["lives"].shape, (num_envs,))
    np.testing.assert_allclose(info["players"]["env_id"].shape, (num_envs,))
    np.testing.assert_allclose(truncated.shape, (num_envs,))
    while not np.any(done):
      env.send(np.random.randint(6, size=num_envs))
      obs, rew, terminated, truncated, info = env.recv()
      done = np.logical_or(terminated, truncated)
    env.send(np.random.randint(6, size=num_envs))
    obs1, rew1, terminated1, truncated1, info1 = env.recv()
    done1 = np.logical_or(terminated1, truncated1)
    index = np.where(done)[0]
    self.assertTrue(np.all(~done1[index]))

  def test_highlevel_step(self) -> None:
    assert version.parse(gym.__version__) >= version.parse("0.26.0")
    num_envs = 4
    env = make_gym("Pong-v5", num_envs=num_envs)
    self.assertTrue(isinstance(env, gym.Env))
    logging.info(env)
    obs, _ = env.reset()
    # check shape
    self.assertIsInstance(obs, np.ndarray)
    self.assertEqual(obs.dtype, np.uint8)
    obs, rew, terminated, truncated, info = env.step(
      np.random.randint(6, size=num_envs)
    )
    done = np.logical_or(terminated, truncated)
    self.assertIsInstance(obs, np.ndarray)
    self.assertEqual(obs.dtype, np.uint8)
    np.testing.assert_allclose(rew.shape, (num_envs,))
    self.assertEqual(rew.dtype, np.float32)
    np.testing.assert_allclose(done.shape, (num_envs,))
    self.assertEqual(done.dtype, np.bool_)
    self.assertIsInstance(info, dict)
    self.assertEqual(len(info), 6)
    self.assertEqual(info["env_id"].dtype, np.int32)
    self.assertEqual(info["lives"].dtype, np.int32)
    self.assertEqual(info["players"]["env_id"].dtype, np.int32)
    self.assertEqual(truncated.dtype, np.bool_)
    np.testing.assert_allclose(info["env_id"], np.arange(num_envs))
    np.testing.assert_allclose(info["lives"].shape, (num_envs,))
    np.testing.assert_allclose(info["players"]["env_id"].shape, (num_envs,))
    np.testing.assert_allclose(truncated.shape, (num_envs,))
    while not np.any(done):
      obs, rew, terminated, truncated, info = env.step(
        np.random.randint(6, size=num_envs)
      )
      done = np.logical_or(terminated, truncated)
    obs1, rew1, terminated1, truncated1, info1 = env.step(
      np.random.randint(6, size=num_envs)
    )
    done1 = np.logical_or(terminated1, truncated1)
    index = np.where(done)[0]
    self.assertTrue(np.all(~done1[index]))


if __name__ == "__main__":
  absltest.main()
