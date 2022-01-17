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
from typing import no_type_check

import cv2
import numpy as np
from absl.testing import absltest

from envpool.vizdoom import VizdoomEnvSpec, VizdoomGymEnvPool


class _VizdoomEnvPoolBasicTest(absltest.TestCase):

  def test_timelimit(
    self, num_envs: int = 5, max_episode_steps: int = 10
  ) -> None:
    conf = VizdoomEnvSpec.gen_config(
      num_envs=num_envs,
      max_episode_steps=max_episode_steps,
      cfg_path="vizdoom/maps/D1_basic.cfg",
      wad_path="vizdoom/maps/D1_basic.wad",
      use_combined_action=True,
    )
    env = VizdoomGymEnvPool(VizdoomEnvSpec(conf))
    for _ in range(3):
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

  @no_type_check
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
    conf = VizdoomEnvSpec.gen_config(
      num_envs=num_envs,
      cfg_path="vizdoom/maps/D1_basic.cfg",
      wad_path="vizdoom/maps/D1_basic.wad",
      use_combined_action=True,
      img_width=width,
      img_height=height,
    )
    env = VizdoomGymEnvPool(VizdoomEnvSpec(conf))
    assert env.action_space.n == 6
    obs = env.reset().transpose(0, 2, 3, 1)
    action_num = env.action_space.n
    env_id = np.arange(num_envs)
    np.random.seed(0)
    for t in range(step):
      assert obs.shape == (num_envs, height, width, 4), obs.shape
      act = np.random.randint(action_num, size=len(env_id))
      obs_, _, done, info = env.step(act, env_id)
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
        obs[done_id] = env.reset(done_id).transpose(0, 2, 3, 1)

  @no_type_check
  def test_d3_action_space(self) -> None:
    conf = VizdoomEnvSpec.gen_config(
      cfg_path="vizdoom/maps/D3_battle.cfg",
      wad_path="vizdoom/maps/D3_battle.wad",
      use_combined_action=True,
    )
    env = VizdoomGymEnvPool(VizdoomEnvSpec(conf))
    action_num = env.action_space.n
    conf = VizdoomEnvSpec.gen_config(
      cfg_path="vizdoom/maps/D3_battle.cfg",
      wad_path="vizdoom/maps/D3_battle.wad",
      use_combined_action=True,
      force_speed=True,
    )
    env = VizdoomGymEnvPool(VizdoomEnvSpec(conf))
    assert env.action_space.n * 2 == action_num

  @no_type_check
  def test_delta_action_space(self) -> None:
    e = VizdoomGymEnvPool(VizdoomEnvSpec(VizdoomEnvSpec.gen_config(
      cfg_path="vizdoom/maps/deathmatch.cfg",
      wad_path="vizdoom/maps/deathmatch.wad",
      use_combined_action=True,
    )))
    e2 = VizdoomGymEnvPool(VizdoomEnvSpec(VizdoomEnvSpec.gen_config(
      cfg_path="vizdoom/maps/deathmatch.cfg",
      wad_path="vizdoom/maps/deathmatch.wad",
      use_combined_action=True,
      delta_button_config={
        "LOOK_UP_DOWN_DELTA": [11, -10, 10],
      }
    )))
    assert e2.action_space.n == 11 * e.action_space.n
    e3 = VizdoomGymEnvPool(VizdoomEnvSpec(VizdoomEnvSpec.gen_config(
      cfg_path="vizdoom/maps/deathmatch.cfg",
      wad_path="vizdoom/maps/deathmatch.wad",
      use_combined_action=True,
      delta_button_config={
        "MOVE_LEFT_RIGHT_DELTA": [11, -10, 10],
        "LOOK_UP_DOWN_DELTA": [11, -10, 10],
      }
    )))
    assert e3.action_space.n == 121 * e.action_space.n
    e4 = VizdoomGymEnvPool(VizdoomEnvSpec(VizdoomEnvSpec.gen_config(
      cfg_path="vizdoom/maps/deathmatch.cfg",
      wad_path="vizdoom/maps/deathmatch.wad",
      use_combined_action=False,
      delta_button_config={
        "MOVE_LEFT_RIGHT_DELTA": [11, -10, 10],
        "LOOK_UP_DOWN_DELTA": [11, -10, 10],
      }
    )))
    assert e4.action_space.shape[0] == 20

  @no_type_check
  def test_depth_buffer(self) -> None:
    e = VizdoomGymEnvPool(VizdoomEnvSpec(VizdoomEnvSpec.gen_config(
      cfg_path="vizdoom/maps/deathmatch.cfg",
      wad_path="vizdoom/maps/deathmatch.wad",
      use_combined_action=True,
    )))
    assert e.observation_space.shape[0] == 3 * 4
    e.reset()
    assert e.step(np.array([0]), np.array([0]))[0].shape[1] == 3 * 4
    e = VizdoomGymEnvPool(VizdoomEnvSpec(VizdoomEnvSpec.gen_config(
      cfg_path="vizdoom/maps/D1_basic.cfg",
      wad_path="vizdoom/maps/D1_basic.wad",
      use_combined_action=True,
    )))
    assert e.observation_space.shape[0] == 1 * 4
    e.reset()
    assert e.step(np.array([0]), np.array([0]))[0].shape[1] == 1 * 4


if __name__ == "__main__":
  absltest.main()
