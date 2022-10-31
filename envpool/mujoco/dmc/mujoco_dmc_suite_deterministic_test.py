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


class _MujocoDmcDeterministicTest(absltest.TestCase):

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

  def test_acrobot(self) -> None:
    obs_keys = ["orientations", "velocity"]
    for task in ["swingup", "swingup_sparse"]:
      self.check("acrobot", task, obs_keys)

  def test_ball_in_cup(self) -> None:
    obs_keys = ["position", "velocity"]
    for task in ["catch"]:
      self.check(
        "ball_in_cup",
        task,
        obs_keys,
        # https://github.com/sail-sg/envpool/pull/124#issuecomment-1127860698
        blacklist=["velocity"],
      )

  def test_cartpole(self) -> None:
    obs_keys = ["position", "velocity"]
    for task in [
      "balance", "balance_sparse", "swingup", "swingup_sparse", "two_poles",
      "three_poles"
    ]:
      self.check("cartpole", task, obs_keys)

  def test_cheetah(self) -> None:
    obs_keys = ["position", "velocity"]
    for task in ["run"]:
      self.check("cheetah", task, obs_keys)

  def test_finger(self) -> None:
    obs_keys = ["position", "velocity", "touch"]
    for task in ["spin"]:
      self.check("finger", task, obs_keys)
    obs_keys += ["target_position", "dist_to_target"]
    for task in ["turn_easy", "turn_hard"]:
      self.check("finger", task, obs_keys)

  def test_fish(self) -> None:
    obs_keys = ["joint_angles", "upright", "velocity"]
    for task in ["swim", "upright"]:
      self.check(
        "fish",
        task,
        obs_keys + (["target"] if task == "swim" else []),
        blacklist=["joint_angles"],
      )

  def test_hopper(self) -> None:
    obs_keys = ["position", "velocity", "touch"]
    for task in ["stand", "hop"]:
      self.check("hopper", task, obs_keys)

  def test_humanoid(self) -> None:
    obs_keys = [
      "joint_angles", "head_height", "extremities", "torso_vertical",
      "com_velocity", "velocity"
    ]
    for task in ["stand", "walk", "run"]:
      self.check("humanoid", task, obs_keys)
    obs_keys = ["position", "velocity"]
    for task in ["run_pure_state"]:
      self.check("humanoid", task, obs_keys)

  def test_manipulator(self) -> None:
    obs_keys = [
      "arm_pos", "arm_vel", "touch", "hand_pos", "object_pos", "object_vel",
      "target_pos"
    ]
    for task in ["bring_ball", "bring_peg", "insert_ball", "insert_peg"]:
      self.check("manipulator", task, obs_keys)

  def test_pendulum(self) -> None:
    obs_keys = ["orientation", "velocity"]
    for task in ["swingup"]:
      self.check("pendulum", task, obs_keys)

  def test_point_mass(self) -> None:
    obs_keys = ["position", "velocity"]
    for task in ["easy", "hard"]:
      self.check(
        "point_mass",
        task,
        obs_keys,
        blacklist=["velocity"] if task == "easy" else [],
      )

  def test_reacher(self) -> None:
    obs_keys = ["position", "to_target", "velocity"]
    for task in ["easy", "hard"]:
      self.check("reacher", task, obs_keys)

  def test_swimmer(self) -> None:
    obs_keys = ["joints", "to_target", "body_velocities"]
    for task in ["swimmer6", "swimmer15"]:
      self.check("swimmer", task, obs_keys)

  def test_walker(self) -> None:
    obs_keys = ["orientations", "height", "velocity"]
    for task in ["run", "stand", "walk"]:
      self.check("walker", task, obs_keys)


if __name__ == "__main__":
  absltest.main()
