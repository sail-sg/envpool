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
"""Unit tests for Mujoco gym deterministic check."""

import numpy as np
from absl.testing import absltest

from envpool.registration import make, make_gym, make_spec


class _MujocoGymDeterministicTest(absltest.TestCase):
    def check(self, task_id: str, num_envs: int = 4) -> None:
        env0 = make_gym(task_id, num_envs=num_envs, seed=0)
        env1 = make_gym(task_id, num_envs=num_envs, seed=0)
        env2 = make_gym(task_id, num_envs=num_envs, seed=1)
        act_space = env0.action_space
        eps = np.finfo(np.float32).eps
        obs_space = env0.observation_space
        obs_min, obs_max = obs_space.low - eps, obs_space.high + eps
        for _ in range(3000):
            action = np.array([act_space.sample() for _ in range(num_envs)])
            obs0 = env0.step(action)[0]
            obs1 = env1.step(action)[0]
            obs2 = env2.step(action)[0]
            np.testing.assert_allclose(obs0, obs1)
            self.assertFalse(np.allclose(obs0, obs2))
            self.assertTrue(np.all(obs_min <= obs0), obs0)
            self.assertTrue(np.all(obs_min <= obs2), obs2)
            self.assertTrue(np.all(obs0 <= obs_max), obs0)
            self.assertTrue(np.all(obs2 <= obs_max), obs2)

    def test_half_cheetah_supports_xml_file(self) -> None:
        xml_file = "half_cheetah_envpool.xml"
        spec = make_spec("HalfCheetah-v5", xml_file=xml_file)
        self.assertEqual(spec.config.xml_file, xml_file)

        env = make("HalfCheetah-v5", env_type="gym", xml_file=xml_file)
        env.reset()
        env.close()

    def test_half_cheetah_frame_stack(self) -> None:
        spec = make_spec("HalfCheetah-v5", frame_stack=4)
        self.assertEqual(spec.observation_space.shape, (4, 17))

        env = make_gym("HalfCheetah-v5", num_envs=1, seed=0, frame_stack=4)
        try:
            obs, _ = env.reset()
            self.assertEqual(env.observation_space.shape, (4, 17))
            self.assertEqual(obs.shape, (1, 4, 17))
            np.testing.assert_allclose(obs[0, 0], obs[0, 1])
            np.testing.assert_allclose(obs[0, 1], obs[0, 2])
            np.testing.assert_allclose(obs[0, 2], obs[0, 3])

            action = np.zeros((1, *env.action_space.shape), dtype=np.float64)
            next_obs = env.step(action)[0]
            self.assertEqual(next_obs.shape, (1, 4, 17))
            np.testing.assert_allclose(next_obs[0, :-1], obs[0, 1:])
            self.assertFalse(np.allclose(next_obs[0, -1], obs[0, -1]))
        finally:
            env.close()

    def test_half_cheetah_frame_stack_one_matches_default(self) -> None:
        default_spec = make_spec("HalfCheetah-v5")
        frame_stack_spec = make_spec("HalfCheetah-v5", frame_stack=1)
        self.assertEqual(default_spec.observation_space.shape, (17,))
        self.assertEqual(
            frame_stack_spec.observation_space.shape,
            default_spec.observation_space.shape,
        )

        env0 = make_gym("HalfCheetah-v5", num_envs=1, seed=0)
        env1 = make_gym("HalfCheetah-v5", num_envs=1, seed=0, frame_stack=1)
        try:
            obs0, _ = env0.reset()
            obs1, _ = env1.reset()
            self.assertEqual(obs0.shape, (1, 17))
            self.assertEqual(obs1.shape, obs0.shape)
            self.assertEqual(
                env1.observation_space.shape, env0.observation_space.shape
            )
            np.testing.assert_allclose(obs0, obs1)

            action = np.zeros((1, *env0.action_space.shape), dtype=np.float64)
            next_obs0 = env0.step(action)[0]
            next_obs1 = env1.step(action)[0]
            self.assertEqual(next_obs1.shape, next_obs0.shape)
            np.testing.assert_allclose(next_obs0, next_obs1)
        finally:
            env0.close()
            env1.close()

    def test_ant(self) -> None:
        self.check("Ant-v5")

    def test_half_cheetah(self) -> None:
        self.check("HalfCheetah-v5")

    def test_hopper(self) -> None:
        self.check("Hopper-v5")

    def test_humanoid(self) -> None:
        self.check("Humanoid-v5")

    def test_humanoid_standup(self) -> None:
        self.check("HumanoidStandup-v5")

    def test_inverted_double_pendulum(self) -> None:
        self.check("InvertedDoublePendulum-v5")

    def test_inverted_pendulum(self) -> None:
        self.check("InvertedPendulum-v5")

    def test_pusher(self) -> None:
        self.check("Pusher-v5")

    def test_reacher(self) -> None:
        self.check("Reacher-v5")

    def test_swimmer(self) -> None:
        self.check("Swimmer-v5")

    def test_walker2d(self) -> None:
        self.check("Walker2d-v5")


if __name__ == "__main__":
    absltest.main()
