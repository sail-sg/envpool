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


class _MujocoDmcDeterministicTest(absltest.TestCase):
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
                        low=act_spec.minimum,
                        high=act_spec.maximum,
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
            "balance",
            "balance_sparse",
            "swingup",
            "swingup_sparse",
            "two_poles",
            "three_poles",
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
            "joint_angles",
            "head_height",
            "extremities",
            "torso_vertical",
            "com_velocity",
            "velocity",
        ]
        for task in ["stand", "walk", "run"]:
            self.check("humanoid", task, obs_keys)
        obs_keys = ["position", "velocity"]
        for task in ["run_pure_state"]:
            self.check("humanoid", task, obs_keys)

    def test_manipulator(self) -> None:
        obs_keys = [
            "arm_pos",
            "arm_vel",
            "touch",
            "hand_pos",
            "object_pos",
            "object_vel",
            "target_pos",
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

    def test_walker_frame_stack(self) -> None:
        env = make_dm("WalkerWalk-v1", num_envs=1, seed=0, frame_stack=4)
        try:
            obs_spec = env.observation_spec()
            self.assertEqual(obs_spec.orientations.shape, (4, 14))
            self.assertEqual(obs_spec.height.shape, (4,))
            self.assertEqual(obs_spec.velocity.shape, (4, 9))

            ts0 = env.reset()
            np.testing.assert_allclose(
                ts0.observation.orientations[0, 0],
                ts0.observation.orientations[0, 1],
            )
            np.testing.assert_allclose(
                ts0.observation.height[0, 0],
                ts0.observation.height[0, 1],
            )
            np.testing.assert_allclose(
                ts0.observation.velocity[0, 0],
                ts0.observation.velocity[0, 1],
            )

            action_spec = env.action_spec()
            action = np.zeros((1,) + action_spec.shape, dtype=action_spec.dtype)
            ts1 = env.step(action)
            np.testing.assert_allclose(
                ts0.observation.orientations[0, 1:],
                ts1.observation.orientations[0, :-1],
            )
            np.testing.assert_allclose(
                ts0.observation.height[0, 1:],
                ts1.observation.height[0, :-1],
            )
            np.testing.assert_allclose(
                ts0.observation.velocity[0, 1:],
                ts1.observation.velocity[0, :-1],
            )
        finally:
            env.close()

    def test_walker_frame_stack_one_matches_default(self) -> None:
        env0 = make_dm("WalkerWalk-v1", num_envs=1, seed=0)
        env1 = make_dm("WalkerWalk-v1", num_envs=1, seed=0, frame_stack=1)
        try:
            obs_spec0 = env0.observation_spec()
            obs_spec1 = env1.observation_spec()
            self.assertEqual(obs_spec0.orientations.shape, (14,))
            self.assertEqual(
                obs_spec0.orientations.shape, obs_spec1.orientations.shape
            )
            self.assertEqual(obs_spec0.height.shape, obs_spec1.height.shape)
            self.assertEqual(obs_spec0.velocity.shape, obs_spec1.velocity.shape)

            ts0 = env0.reset()
            ts1 = env1.reset()
            np.testing.assert_allclose(
                ts0.observation.orientations, ts1.observation.orientations
            )
            np.testing.assert_allclose(
                ts0.observation.height, ts1.observation.height
            )
            np.testing.assert_allclose(
                ts0.observation.velocity, ts1.observation.velocity
            )

            action_spec = env0.action_spec()
            action = np.zeros((1,) + action_spec.shape, dtype=action_spec.dtype)
            ts0 = env0.step(action)
            ts1 = env1.step(action)
            np.testing.assert_allclose(
                ts0.observation.orientations, ts1.observation.orientations
            )
            np.testing.assert_allclose(
                ts0.observation.height, ts1.observation.height
            )
            np.testing.assert_allclose(
                ts0.observation.velocity, ts1.observation.velocity
            )
        finally:
            env0.close()
            env1.close()


if __name__ == "__main__":
    absltest.main()
