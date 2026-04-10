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
        obs_keys: list[str],
        blacklist: list[str] | None = None,
        num_envs: int = 4,
        num_steps: int = 1000,
    ) -> None:
        domain_name = "".join([
            g[:1].upper() + g[1:] for g in domain.split("_")
        ])
        task_name = "".join([g[:1].upper() + g[1:] for g in task.split("_")])
        task_id = f"{domain_name}{task_name}-v1"
        np.random.seed(0)
        env0 = make_dm(task_id, num_envs=num_envs, seed=0)
        env1 = make_dm(task_id, num_envs=num_envs, seed=0)
        env2 = make_dm(task_id, num_envs=num_envs, seed=1)
        act_spec = env0.action_spec()
        action_min = act_spec.minimum
        action_max = act_spec.maximum
        if domain == "lqr":
            action_min = -1.0
            action_max = 1.0
        try:
            ts0 = env0.reset()
            ts1 = env1.reset()
            ts2 = env2.reset()
            for k in obs_keys:
                np.testing.assert_allclose(
                    getattr(ts0.observation, k),
                    getattr(ts1.observation, k),
                )
            seed1_is_allclose = all(
                np.allclose(
                    getattr(ts0.observation, k),
                    getattr(ts2.observation, k),
                )
                for k in obs_keys
                if not (blacklist and k in blacklist)
            )
            last_step = -1
            for _ in range(num_steps):
                last_step += 1
                action = np.array([
                    np.random.uniform(
                        low=action_min,
                        high=action_max,
                        size=act_spec.shape,
                    )
                    for _ in range(num_envs)
                ])
                ts0 = env0.step(action)
                ts1 = env1.step(action)
                ts2 = env2.step(action)
                np.testing.assert_array_equal(ts0.step_type, ts1.step_type)
                np.testing.assert_allclose(ts0.reward, ts1.reward)
                np.testing.assert_allclose(ts0.discount, ts1.discount)
                obs0 = ts0.observation
                obs1 = ts1.observation
                obs2 = ts2.observation
                comparable_obs = False
                for k in obs_keys:
                    o0 = getattr(obs0, k)
                    o1 = getattr(obs1, k)
                    o2 = getattr(obs2, k)
                    np.testing.assert_allclose(o0, o1)
                    if blacklist and k in blacklist:
                        continue
                    if (
                        np.abs(o0).sum() > 0
                        and ts0.step_type[0] != dm_env.StepType.FIRST
                    ):
                        comparable_obs = True
                        seed1_is_allclose &= np.allclose(o0, o2)
                if comparable_obs:
                    seed1_is_allclose &= np.allclose(ts0.reward, ts2.reward)
            self.assertFalse(seed1_is_allclose, (domain, task, last_step))
        finally:
            env0.close()
            env1.close()
            env2.close()

    def test_humanoid_CMU(self) -> None:
        obs_keys = [
            "joint_angles",
            "head_height",
            "extremities",
            "torso_vertical",
            "com_velocity",
            "velocity",
        ]
        for task in ["stand", "walk", "run"]:
            self.check("humanoid_CMU", task, obs_keys)

    def test_dog(self) -> None:
        obs_keys = [
            "joint_angles",
            "joint_velocites",
            "torso_pelvis_height",
            "z_projection",
            "torso_com_velocity",
            "inertial_sensors",
            "foot_forces",
            "touch_sensors",
            "actuator_state",
        ]
        for task in ["stand", "walk", "trot", "run"]:
            self.check("dog", task, obs_keys)
        self.check("dog", "fetch", obs_keys + ["ball_state", "target_position"])

    def test_lqr(self) -> None:
        for task in ["lqr_2_1", "lqr_6_2"]:
            self.check("lqr", task, ["position", "velocity"])

    def test_quadruped(self) -> None:
        obs_keys = [
            "egocentric_state",
            "torso_velocity",
            "torso_upright",
            "imu",
            "force_torque",
        ]
        for task in ["walk", "run"]:
            self.check(
                "quadruped",
                task,
                obs_keys,
                blacklist=["egocentric_state"],
            )
        self.check(
            "quadruped",
            "escape",
            obs_keys + ["origin", "rangefinder"],
            blacklist=["egocentric_state"],
        )
        self.check(
            "quadruped",
            "fetch",
            obs_keys + ["ball_state", "target_position"],
            blacklist=["egocentric_state"],
        )

    def test_stacker(self) -> None:
        obs_keys = [
            "arm_pos",
            "arm_vel",
            "touch",
            "hand_pos",
            "box_pos",
            "box_vel",
            "target_pos",
        ]
        for task in ["stack_2", "stack_4"]:
            self.check("stacker", task, obs_keys, blacklist=["box_vel"])


if __name__ == "__main__":
    absltest.main()
