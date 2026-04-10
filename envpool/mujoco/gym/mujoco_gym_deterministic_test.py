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

from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.mujoco.gym.registration as reg
from envpool.registration import make, make_gym, make_spec

_TASK_IDS = tuple(
    sorted(
        f"{task}-{version}"
        for task, versions, _ in reg.gym_mujoco_envs
        for version in versions
        if version == "v5"
    )
)


def _max_episode_steps(task_id: str) -> int:
    return int(make_spec(task_id).config.max_episode_steps)


class _MujocoGymDeterministicTest(absltest.TestCase):
    def _assert_info_equal(
        self, info0: dict[str, Any], info1: dict[str, Any]
    ) -> None:
        self.assertEqual(info0.keys(), info1.keys())
        for key in info0:
            self._assert_value_equal(info0[key], info1[key], f"info[{key}]")

    def _assert_value_equal(self, value0: Any, value1: Any, label: str) -> None:
        if isinstance(value0, dict):
            self.assertIsInstance(value1, dict)
            self.assertEqual(value0.keys(), value1.keys())
            for key in value0:
                self._assert_value_equal(
                    value0[key],
                    value1[key],
                    f"{label}[{key}]",
                )
        else:
            arr0 = np.asarray(value0)
            arr1 = np.asarray(value1)
            if arr0.dtype == object or arr1.dtype == object:
                self.assertEqual(arr0.shape, arr1.shape, label)
                for index in np.ndindex(arr0.shape):
                    self._assert_value_equal(
                        arr0[index],
                        arr1[index],
                        f"{label}{index}",
                    )
                return
            np.testing.assert_allclose(value0, value1, err_msg=label)

    def check(
        self,
        task_id: str,
        num_envs: int = 4,
        total: int | None = None,
        action_seed: int = 1,
    ) -> None:
        if total is None:
            total = _max_episode_steps(task_id)
        env0 = make_gym(task_id, num_envs=num_envs, seed=0)
        env1 = make_gym(task_id, num_envs=num_envs, seed=0)
        env2 = make_gym(task_id, num_envs=num_envs, seed=1)
        act_space = env0.action_space
        act_space.seed(action_seed)
        eps = np.finfo(np.float32).eps
        obs_space = env0.observation_space
        obs_min, obs_max = obs_space.low - eps, obs_space.high + eps
        try:
            obs0, info0 = env0.reset()
            obs1, info1 = env1.reset()
            obs2, _ = env2.reset()
            np.testing.assert_allclose(obs0, obs1)
            self._assert_info_equal(info0, info1)
            differs = not np.allclose(obs0, obs2)
            for _ in range(total):
                action = np.array([act_space.sample() for _ in range(num_envs)])
                obs0, rew0, term0, trunc0, info0 = env0.step(action)
                obs1, rew1, term1, trunc1, info1 = env1.step(action)
                obs2, rew2, term2, trunc2, _ = env2.step(action)
                np.testing.assert_allclose(obs0, obs1)
                np.testing.assert_allclose(rew0, rew1)
                np.testing.assert_array_equal(term0, term1)
                np.testing.assert_array_equal(trunc0, trunc1)
                self._assert_info_equal(info0, info1)
                differs = (
                    differs
                    or not np.allclose(obs0, obs2)
                    or not np.allclose(rew0, rew2)
                    or not np.array_equal(term0, term2)
                    or not np.array_equal(trunc0, trunc2)
                )
                self.assertTrue(np.all(obs_min <= obs0), obs0)
                self.assertTrue(np.all(obs_min <= obs2), obs2)
                self.assertTrue(np.all(obs0 <= obs_max), obs0)
                self.assertTrue(np.all(obs2 <= obs_max), obs2)
            self.assertTrue(
                differs, msg=f"expected seed divergence for {task_id}"
            )
        finally:
            env0.close()
            env1.close()
            env2.close()

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

    def test_registered_tasks(self) -> None:
        for task_id in _TASK_IDS:
            with self.subTest(task_id=task_id):
                self.check(task_id)


if __name__ == "__main__":
    absltest.main()
