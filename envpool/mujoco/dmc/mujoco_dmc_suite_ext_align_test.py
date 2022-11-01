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

from typing import Any, List

import dm_env
import numpy as np
from absl import logging
from absl.testing import absltest
from dm_control import suite

import envpool.mujoco.dmc.registration  # noqa: F401
from envpool.registration import make_dm


class _MujocoDmcSuiteExtAlignTest(absltest.TestCase):

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

  def reset_state(
    self, env: dm_env.Environment, ts: dm_env.TimeStep, domain: str, task: str
  ) -> None:
    # manually reset, mimic initialize_episode
    with env.physics.reset_context():
      env.physics.data.qpos = ts.observation.qpos0[0]
      if domain in ["humanoid_CMU"]:
        env.physics.after_reset()

  def sample_action(self, action_spec: dm_env.specs.Array) -> np.ndarray:
    return np.random.uniform(
      low=action_spec.minimum,
      high=action_spec.maximum,
      size=action_spec.shape,
    )

  def run_align_check(
    self, env0: dm_env.Environment, env1: Any, domain: str, task: str
  ) -> None:
    logging.info(f"align check for {domain} {task}")
    obs_spec, action_spec = env0.observation_spec(), env0.action_spec()
    for i in range(3):
      np.random.seed(i)
      env0.reset()
      a = self.sample_action(action_spec)
      ts = env1.reset(np.array([0]))
      self.reset_state(env0, ts, domain, task)
      logging.info(f"reset qpos {ts.observation.qpos0[0]}")
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
        np.testing.assert_allclose(ts0.reward, ts1.reward[0], atol=1e-8)
        np.testing.assert_allclose(ts0.discount, ts1.discount[0])

  def run_align_check_entry(self, domain: str, tasks: List[str]) -> None:
    domain_name = "".join([g[:1].upper() + g[1:] for g in domain.split("_")])
    for task in tasks:
      task_name = "".join([g[:1].upper() + g[1:] for g in task.split("_")])
      env0 = suite.load(domain, task)
      env1 = make_dm(f"{domain_name}{task_name}-v1")
      self.run_space_check(env0, env1)
      self.run_align_check(env0, env1, domain, task)

  def test_humanoid_CMU(self) -> None:
    self.run_align_check_entry("humanoid_CMU", ["stand", "run"])


if __name__ == "__main__":
  absltest.main()
