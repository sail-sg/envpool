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

from typing import Any, List

import numpy as np
from absl.testing import absltest

from envpool.mujoco import DmcHopperDMEnvPool, DmcHopperEnvSpec


class _MujocoDmcDeterministicTest(absltest.TestCase):

  def check(
    self,
    spec_cls: Any,
    envpool_cls: Any,
    task: str,
    obs_keys: List[str],
    num_envs: int = 4,
  ) -> None:
    env0 = envpool_cls(
      spec_cls(spec_cls.gen_config(num_envs=num_envs, seed=0, task_name=task))
    )
    env1 = envpool_cls(
      spec_cls(spec_cls.gen_config(num_envs=num_envs, seed=0, task_name=task))
    )
    env2 = envpool_cls(
      spec_cls(spec_cls.gen_config(num_envs=num_envs, seed=1, task_name=task))
    )
    act_spec = env0.action_spec()
    for _ in range(3000):
      action = np.array(
        [
          np.random.uniform(
            low=act_spec.minimum, high=act_spec.maximum, size=act_spec.shape
          ) for _ in range(num_envs)
        ]
      )
      obs0 = env0.step(action).observation
      obs1 = env1.step(action).observation
      obs2 = env2.step(action).observation
      for k in obs_keys:
        o0 = getattr(obs0, k)
        o1 = getattr(obs1, k)
        o2 = getattr(obs2, k)
        np.testing.assert_allclose(o0, o1)
        if np.abs(o0).sum() > 0:
          self.assertFalse(np.allclose(o0, o2))

  def test_hopper(self) -> None:
    obs_keys = ["position", "velocity", "touch"]
    self.check(DmcHopperEnvSpec, DmcHopperDMEnvPool, "stand", obs_keys)
    self.check(DmcHopperEnvSpec, DmcHopperDMEnvPool, "hop", obs_keys)


if __name__ == "__main__":
  absltest.main()
