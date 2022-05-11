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

from typing import Any, no_type_check

import dm_env
import numpy as np
from absl import logging
from absl.testing import absltest
from dm_control import suite

from envpool.mujoco import DmcHopperDMEnvPool, DmcHopperEnvSpec


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
  def reset_state(self, env: dm_env.Environment, ts: dm_env.TimeStep) -> None:
    # manually reset
    with env.physics.reset_context():
      env.physics.data.qpos = ts.observation.qpos0[0]

  def sample_action(self, action_spec: dm_env.specs.Array) -> np.ndarray:
    return np.random.uniform(
      low=action_spec.minimum,
      high=action_spec.maximum,
      size=action_spec.shape,
    )

  def run_align_check(self, env0: dm_env.Environment, env1: Any) -> None:
    logging.info(f"align check for {env1.__class__.__name__}")
    obs_spec, action_spec = env0.observation_spec(), env0.action_spec()
    for i in range(5):
      np.random.seed(i)
      env0.reset()
      a = self.sample_action(action_spec)
      ts = env1.reset(np.array([0]))
      self.reset_state(env0, ts)
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
        np.testing.assert_allclose(ts0.step_type, ts1.step_type[0])
        np.testing.assert_allclose(ts0.reward, ts1.reward[0])
        np.testing.assert_allclose(ts0.discount, ts1.discount[0])
        o0, o1 = ts0.observation, ts1.observation
        for k in obs_spec:
          np.testing.assert_allclose(o0[k], getattr(o1, k)[0])

  def test_hopper(self) -> None:
    env0 = suite.load("hopper", "stand")
    env1 = DmcHopperDMEnvPool(
      DmcHopperEnvSpec(DmcHopperEnvSpec.gen_config(task_name="stand"))
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1)

    env0 = suite.load("hopper", "hop")
    env1 = DmcHopperDMEnvPool(
      DmcHopperEnvSpec(DmcHopperEnvSpec.gen_config(task_name="hop"))
    )
    self.run_space_check(env0, env1)
    self.run_align_check(env0, env1)


if __name__ == "__main__":
  absltest.main()
