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
"""Unit tests for Mujoco dm_control suite align check."""

from typing import Any, List, no_type_check

import dm_env
import numpy as np
from absl import logging
from absl.testing import absltest
from dm_control import suite

from envpool.mujoco import (
  DmcCheetahDMEnvPool,
  DmcCheetahEnvSpec,
  DmcFingerDMEnvPool,
  DmcFingerEnvSpec,
  DmcHopperDMEnvPool,
  DmcHopperEnvSpec,
  DmcReacherDMEnvPool,
  DmcReacherEnvSpec,
  DmcWalkerDMEnvPool,
  DmcWalkerEnvSpec,
)


class _MujocoDmcAlignTest(absltest.TestCase):

  @no_type_check
  def run_space_check(self, env0: dm_env.Environment, env1: Any) -> None:
    """Check observation_spec() and action_spec()."""
    obs0, obs1 = env0.observation_spec(), env1.observation_spec()
    for k in obs0:
      self.assertTrue(hasattr(obs1, k))
      np.testing.assert_allclose(obs0[k].shape, getattr(obs1, k).shape)
    act0, act1 = env0.action_spec(), env1.action_spec()
    np.testing.assert_allclose(act0.shape, act1.shape)
    np.testing.assert_allclose(act0.minimum, act1.minimum)
    np.testing.assert_allclose(act0.maximum, act1.maximum)

  @no_type_check
  def reset_state(
    self, env: dm_env.Environment, ts: dm_env.TimeStep, domain: str, task: str
  ) -> None:
    # manually reset, mimic initialize_episode
    with env.physics.reset_context():
      env.physics.data.qpos = ts.observation.qpos0[0]
      if domain == "cheetah":
        for _ in range(200):
          env.physics.step()
        env.physics.data.time = 0
      elif domain == "reacher":
        target = ts.observation.target[0]
        env.physics.named.model.geom_pos["target", "x"] = target[0]
        env.physics.named.model.geom_pos["target", "y"] = target[1]
      elif domain == "finger" and task == "spin":
        env.physics.named.model.site_rgba['target',
                                          3] = ts.observation.rgba[0][0]
        env.physics.named.model.site_rgba['tip', 3] = ts.observation.rgba[0][1]
        dof = ts.observation.dof_damping[0]
        env.physics.named.model.dof_damping['hinge'] = dof
      elif domain == "finger" and (task == "turn_easy" or task == "turn_hard"):
        env.physics.named.model.site_pos['target',
                                         ['x', 'z']] = ts.observation.target
        env.physics.named.model.site_size['target',
                                          0] = ts.observation.site_size

  def sample_action(self, action_spec: dm_env.specs.Array) -> np.ndarray:
    return np.random.uniform(
      low=action_spec.minimum,
      high=action_spec.maximum,
      size=action_spec.shape,
    )

  def run_align_check(
    self, env0: dm_env.Environment, env1: Any, domain: str, task: str
  ) -> None:
    logging.info(f"align check for {env1.__class__.__name__}")
    obs_spec, action_spec = env0.observation_spec(), env0.action_spec()
    for i in range(5):
      np.random.seed(i)
      env0.reset()
      a = self.sample_action(action_spec)
      ts = env1.reset(np.array([0]))
      self.reset_state(env0, ts, domain, task)
      logging.info(f'reset qpos {ts.observation.qpos0[0]}')
      cnt = 0
      done = False
      while not done:
        cnt += 1
        a = self.sample_action(action_spec)
        # logging.info(f"{cnt} {a}")
        ts0 = env0.step(a)
        ts1 = env1.step(np.array([a]), np.array([0]))
        done = ts0.step_type == dm_env.StepType.LAST
        o0, o1 = ts0.observation, ts1.observation
        for k in obs_spec:
          np.testing.assert_allclose(o0[k], getattr(o1, k)[0])
        np.testing.assert_allclose(ts0.step_type, ts1.step_type[0])
        np.testing.assert_allclose(ts0.reward, ts1.reward[0])
        np.testing.assert_allclose(ts0.discount, ts1.discount[0])

  def run_align_check_entry(
    self, domain: str, tasks: List[str], spec_cls: Any, envpool_cls: Any
  ) -> None:
    for task in tasks:
      env0 = suite.load(domain, task)
      env1 = envpool_cls(spec_cls(spec_cls.gen_config(task_name=task)))
      self.run_space_check(env0, env1)
      self.run_align_check(env0, env1, domain, task)

  def test_cheetah(self) -> None:
    self.run_align_check_entry(
      "cheetah", ["run"], DmcCheetahEnvSpec, DmcCheetahDMEnvPool
    )

  def test_finger(self) -> None:
    self.run_align_check_entry(
      "finger", ["spin", "turn_easy", "turn_hard"], DmcFingerEnvSpec,
      DmcFingerDMEnvPool
    )

  def test_hopper(self) -> None:
    self.run_align_check_entry(
      "hopper", ["hop", "stand"], DmcHopperEnvSpec, DmcHopperDMEnvPool
    )

  def test_reacher(self) -> None:
    self.run_align_check_entry(
      "reacher", ["easy", "hard"], DmcReacherEnvSpec, DmcReacherDMEnvPool
    )

  def test_walker(self) -> None:
    self.run_align_check_entry(
      "walker", ["run", "stand", "walk"], DmcWalkerEnvSpec, DmcWalkerDMEnvPool
    )


if __name__ == "__main__":
  absltest.main()
