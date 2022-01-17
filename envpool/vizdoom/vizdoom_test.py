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

import numpy as np
from absl.testing import absltest

from envpool.vizdoom import VizdoomEnvSpec, VizdoomGymEnvPool


class _VizdoomEnvPoolTest(absltest.TestCase):

  def test_timelimit(
    self, num_envs: int = 5, max_episode_steps: int = 10
  ) -> None:
    conf = VizdoomEnvSpec.gen_config(
      num_envs=num_envs,
      max_episode_steps=max_episode_steps,
      cfg_path="vizdoom/maps/D1_basic.cfg",
      wad_path="vizdoom/maps/D1_basic.wad",
      use_raw_action=False,
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


if __name__ == "__main__":
  absltest.main()
