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

from typing import Any, List, Optional

import dm_env
import numpy as np
from absl.testing import absltest

from envpool.mujoco.dmc import (
  DmcAcrobotDMEnvPool,
  DmcAcrobotEnvSpec,
  DmcBallInCupDMEnvPool,
  DmcBallInCupEnvSpec,
  DmcCheetahDMEnvPool,
  DmcCheetahEnvSpec,
  DmcFingerDMEnvPool,
  DmcFingerEnvSpec,
  DmcHopperDMEnvPool,
  DmcHopperEnvSpec,
  DmcPendulumDMEnvPool,
  DmcPendulumEnvSpec,
  DmcPointMassDMEnvPool,
  DmcPointMassEnvSpec,
  DmcReacherDMEnvPool,
  DmcReacherEnvSpec,
  DmcWalkerDMEnvPool,
  DmcWalkerEnvSpec,
)


class _MujocoDmcDeterministicTest(absltest.TestCase):

  def check(
    self,
    spec_cls: Any,
    envpool_cls: Any,
    task: str,
    obs_keys: List[str],
    blacklist: Optional[List[str]] = None,
    num_envs: int = 4,
  ) -> None:
    np.random.seed(0)
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
          self.assertFalse(np.allclose(o0, o2), (k, o0, o2))

  def test_acrobot(self) -> None:
    obs_keys = ["orientations", "velocity"]
    for task in ["swingup", "swingup_sparse"]:
      self.check(DmcAcrobotEnvSpec, DmcAcrobotDMEnvPool, task, obs_keys)

  def test_ball_in_cup(self) -> None:
    obs_keys = ["position", "velocity"]
    for task in ["catch"]:
      self.check(
        DmcBallInCupEnvSpec,
        DmcBallInCupDMEnvPool,
        task,
        obs_keys,
        # https://github.com/sail-sg/envpool/pull/124#issuecomment-1127860698
        blacklist=["velocity"],
      )

  def test_cheetah(self) -> None:
    obs_keys = ["position", "velocity"]
    for task in ["run"]:
      self.check(DmcCheetahEnvSpec, DmcCheetahDMEnvPool, task, obs_keys)

  def test_finger(self) -> None:
    obs_keys = ["position", "velocity", "touch"]
    for task in ["spin"]:
      self.check(DmcFingerEnvSpec, DmcFingerDMEnvPool, task, obs_keys)
    obs_keys += ["target_position", "dist_to_target"]
    for task in ["turn_easy", "turn_hard"]:
      self.check(DmcFingerEnvSpec, DmcFingerDMEnvPool, task, obs_keys)

  def test_hopper(self) -> None:
    obs_keys = ["position", "velocity", "touch"]
    for task in ["stand", "hop"]:
      self.check(DmcHopperEnvSpec, DmcHopperDMEnvPool, task, obs_keys)

  def test_pendulum(self) -> None:
    obs_keys = ["orientation", "velocity"]
    for task in ["swingup"]:
      self.check(DmcPendulumEnvSpec, DmcPendulumDMEnvPool, task, obs_keys)

  def test_point_mass(self) -> None:
    obs_keys = ["position", "velocity"]
    for task in ["easy", "hard"]:
      self.check(
        DmcPointMassEnvSpec,
        DmcPointMassDMEnvPool,
        task,
        obs_keys,
        blacklist=["velocity"] if task == "easy" else [],
      )

  def test_reacher(self) -> None:
    obs_keys = ["position", "to_target", "velocity"]
    for task in ["easy", "hard"]:
      self.check(DmcReacherEnvSpec, DmcReacherDMEnvPool, task, obs_keys)

  def test_walker(self) -> None:
    obs_keys = ["orientations", "height", "velocity"]
    for task in ["run", "stand", "walk"]:
      self.check(DmcWalkerEnvSpec, DmcWalkerDMEnvPool, task, obs_keys)


if __name__ == "__main__":
  absltest.main()
