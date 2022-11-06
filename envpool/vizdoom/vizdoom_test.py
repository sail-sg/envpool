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
"""Unit tests for vizdoom environments."""

import os

import cv2
import numpy as np
from absl.testing import absltest

import envpool.vizdoom.registration  # noqa: F401
from envpool.registration import make_dm, make_gym


class _VizdoomEnvPoolBasicTest(absltest.TestCase):

  def get_path(self, path: str) -> str:
    return os.path.join("envpool", "vizdoom", "maps", path)

  def test_timelimit(
    self, num_envs: int = 5, max_episode_steps: int = 10
  ) -> None:
    env = make_gym(
      "D1Basic-v1",
      num_envs=num_envs,
      max_episode_steps=max_episode_steps,
      use_combined_action=True
    )
    for _ in range(3):
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

  def test_hg(
    self,
    num_envs: int = 10,
    step: int = 5000,
    width: int = 160,
    height: int = 120,
    render: bool = False,
  ) -> None:
    if render:
      os.makedirs("img", exist_ok=True)
    env = make_gym(
      "D1Basic-v1",
      num_envs=num_envs,
      use_combined_action=True,
      img_width=width,
      img_height=height
    )
    assert env.action_space.n == 6
    obs, _ = env.reset()
    obs = obs.transpose(0, 2, 3, 1)
    action_num = env.action_space.n
    env_id = np.arange(num_envs)
    np.random.seed(0)
    for t in range(step):
      assert obs.shape == (num_envs, height, width, 4), obs.shape
      act = np.random.randint(action_num, size=len(env_id))
      obs_, _, terminated, truncated, info = env.step(act, env_id)
      done = np.logical_or(terminated, truncated)
      env_id = info["env_id"]
      if render:
        obs[env_id] = obs_.transpose(0, 2, 3, 1)
        obs[env_id[done]] = 255
        obs_all = np.zeros((height, width * num_envs, 3), np.uint8)
        for j in range(num_envs):
          obs_all[:, width * j:width * (j + 1)] = obs[j, ..., :-1]
        cv2.imwrite(f"img/{t}.png", obs_all)
      if np.any(done):  # even though .step can auto reset
        done_id = np.array(info["env_id"])[done]
        obs[done_id] = env.reset(done_id)[0].transpose(0, 2, 3, 1)

  def test_d3_action_space(self) -> None:
    env = make_gym("D3Battle-v1", use_combined_action=True)
    action_num = env.action_space.n
    env = make_gym("D3Battle-v1", use_combined_action=True, force_speed=True)
    assert env.action_space.n * 2 == action_num

  def test_delta_action_space(self) -> None:
    e = make_gym("Deathmatch-v1", use_combined_action=True)
    e2 = make_gym(
      "Deathmatch-v1",
      use_combined_action=True,
      delta_button_config={
        "LOOK_UP_DOWN_DELTA": [11, -10, 10],
      }
    )
    assert e2.action_space.n == 11 * e.action_space.n
    e3 = make_gym(
      "Deathmatch-v1",
      use_combined_action=True,
      delta_button_config={
        "MOVE_LEFT_RIGHT_DELTA": [11, -10, 10],
        "LOOK_UP_DOWN_DELTA": [11, -10, 10],
      }
    )
    assert e3.action_space.n == 121 * e.action_space.n
    e4 = make_gym(
      "Deathmatch-v1",
      use_combined_action=False,
      delta_button_config={
        "MOVE_LEFT_RIGHT_DELTA": [11, -10, 10],
        "LOOK_UP_DOWN_DELTA": [11, -10, 10],
      }
    )
    assert e4.action_space.shape[0] == 20

  def test_obs_space(self) -> None:
    e = make_dm(
      "Deathmatch-v1",
      use_combined_action=True,
    )
    assert e.observation_spec().obs.shape[0] == 3 * 4
    e.reset()
    assert e.step(np.array([0]),
                  np.array([0])).observation.obs.shape[1] == 3 * 4
    e = make_dm("D1Basic-v1", use_combined_action=True)
    assert e.observation_spec().obs.shape[0] == 1 * 4
    e.reset()
    assert e.step(np.array([0]),
                  np.array([0])).observation.obs.shape[1] == 1 * 4


if __name__ == "__main__":
  absltest.main()
