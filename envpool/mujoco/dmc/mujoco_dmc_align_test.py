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
  DmcHumanoidDMEnvPool,
  DmcHumanoidEnvSpec,
  DmcManipulatorDMEnvPool,
  DmcManipulatorEnvSpec,
  DmcPendulumDMEnvPool,
  DmcPendulumEnvSpec,
  DmcPointMassDMEnvPool,
  DmcPointMassEnvSpec,
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
      elif domain in ["finger", "ball_in_cup", "humanoid"]:
        if domain == "finger" and task in ["turn_easy", "turn_hard"]:
          target_angle = ts.observation.target[0][0]
          hinge = env.physics.named.data.xanchor["hinge", ["x", "z"]]
          radius = env.physics.named.model.geom_size["cap1"].sum()
          target_x = hinge[0] + radius * np.sin(target_angle)
          target_z = hinge[1] + radius * np.cos(target_angle)
          env.physics.named.model.site_pos["target",
                                           ["x", "z"]] = target_x, target_z
        env.physics.after_reset()
      elif domain == "point_mass":
        env.physics.model.wrap_prm = ts.observation.wrap_prm[0]
      elif domain == "manipulator":
        # config
        use_peg = "peg" in task
        insert = "insert" in task
        obj = "peg" if use_peg else "ball"
        target = "target_" + obj
        obj_joints = [obj + "_x", obj + "_z", obj + "_y"]
        receptacle = "slot" if use_peg else "cup"
        (
          target_x, target_z, target_angle, init_type, object_x, object_z,
          object_angle, qvel_objx
        ) = ts.observation.random_info[0]
        logging.info(ts.observation.random_info[0])
        p = env.physics.named
        # assign random value from envpool
        if insert:
          p.model.body_pos[receptacle, ["x", "z"]] = target_x, target_z
          p.model.body_quat[receptacle, ["qw", "qy"]] = (
            np.cos(target_angle / 2), np.sin(target_angle / 2)
          )
        p.model.body_pos[target, ["x", "z"]] = target_x, target_z
        p.model.body_quat[target, ["qw", "qy"]] = \
          (np.cos(target_angle / 2), np.sin(target_angle / 2))
        if np.isclose(init_type, 1):
          np.testing.assert_allclose(
            [object_x, object_z, object_angle],
            [target_x, target_z, target_angle]
          )
        elif np.isclose(init_type, 2):
          env.physics.after_reset()
          np.testing.assert_allclose(
            [object_x, object_z], p.data.site_xpos["grasp", ["x", "z"]]
          )
          grasp_direction = p.data.site_xmat["grasp", ["xx", "zx"]]
          np.testing.assert_allclose(
            object_angle,
            np.pi - np.arctan2(grasp_direction[1], grasp_direction[0])
          )
        else:
          p.data.qvel[obj + "_x"] = qvel_objx
        p.data.qpos[obj_joints] = object_x, object_z, object_angle
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

  def run_align_check_entry(
    self, domain: str, tasks: List[str], spec_cls: Any, envpool_cls: Any
  ) -> None:
    for task in tasks:
      env0 = suite.load(domain, task)
      env1 = envpool_cls(spec_cls(spec_cls.gen_config(task_name=task)))
      self.run_space_check(env0, env1)
      self.run_align_check(env0, env1, domain, task)

  def test_acrobot(self) -> None:
    self.run_align_check_entry(
      "acrobot", ["swingup", "swingup_sparse"], DmcAcrobotEnvSpec,
      DmcAcrobotDMEnvPool
    )

  def test_ball_in_cup(self) -> None:
    self.run_align_check_entry(
      "ball_in_cup", ["catch"], DmcBallInCupEnvSpec, DmcBallInCupDMEnvPool
    )

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

  def test_humanoid(self) -> None:
    self.run_align_check_entry(
      "humanoid", ["stand", "walk", "run", "run_pure_state"],
      DmcHumanoidEnvSpec, DmcHumanoidDMEnvPool
    )

  def test_manipulator(self) -> None:
    self.run_align_check_entry(
      "manipulator", ["bring_ball", "bring_peg", "insert_ball", "insert_peg"],
      DmcManipulatorEnvSpec, DmcManipulatorDMEnvPool
    )

  def test_pendulum(self) -> None:
    self.run_align_check_entry(
      "pendulum", ["swingup"], DmcPendulumEnvSpec, DmcPendulumDMEnvPool
    )

  def test_point_mass(self) -> None:
    self.run_align_check_entry(
      "point_mass", ["easy", "hard"], DmcPointMassEnvSpec,
      DmcPointMassDMEnvPool
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
