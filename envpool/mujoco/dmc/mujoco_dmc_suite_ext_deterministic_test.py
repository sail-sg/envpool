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
"""Unit tests for Mujoco dm_control deterministic check."""

from typing import List, Optional

import dm_env
import numpy as np
from absl.testing import absltest

import envpool.mujoco.dmc.registration  # noqa: F401
from envpool.registration import make_dm


class _MujocoDmcSuiteExtDeterministicTest(absltest.TestCase):

  def check(
    self,
    domain: str,
    task: str,
    obs_keys: List[str],
    blacklist: Optional[List[str]] = None,
    num_envs: int = 4,
  ) -> None:
    domain_name = "".join([g[:1].upper() + g[1:] for g in domain.split("_")])
    task_name = "".join([g[:1].upper() + g[1:] for g in task.split("_")])
    task_id = f"{domain_name}{task_name}-v1"
    np.random.seed(0)
    env0 = make_dm(task_id, num_envs=num_envs, seed=0)
    env1 = make_dm(task_id, num_envs=num_envs, seed=0)
    env2 = make_dm(task_id, num_envs=num_envs, seed=1)
    act_spec = env0.action_spec()
    for t in range(3000):
      action = np.array(
        [
          np.random.uniform(
            low=act_spec.minimum, high=act_spec.maximum, size=act_spec.shape
          ) for _ in range(num_envs)
        ]
      )
      ts0 = env0.step(action)
      obs0 = ts0.observation
      obs1 = env1.step(action).observation
      obs2 = env2.step(action).observation
      for k in obs_keys:
        o0 = getattr(obs0, k)
        o1 = getattr(obs1, k)
        o2 = getattr(obs2, k)
        np.testing.assert_allclose(o0, o1)
        if blacklist and k in blacklist:
          continue
        if np.abs(o0).sum() > 0 and ts0.step_type[0] != dm_env.StepType.FIRST:
          self.assertFalse(np.allclose(o0, o2), (t, k, o0, o2))

  def test_humanoid_CMU(self) -> None:
    obs_keys = [
      "joint_angles", "head_height", "extremities", "torso_vertical",
      "com_velocity", "velocity"
    ]
    for task in ["stand", "run"]:
      self.check("humanoid_CMU", task, obs_keys)


if __name__ == "__main__":
  absltest.main()
