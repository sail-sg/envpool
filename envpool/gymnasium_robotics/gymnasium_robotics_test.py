# Copyright 2026 Garena Online Private Limited
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
"""Tests for the Gymnasium-Robotics adapter backend."""

from __future__ import annotations

import sys
import warnings
from typing import Any, cast

import dm_env
import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np
from absl.testing import absltest

import envpool.gymnasium_robotics.registration as robotics_registration
from envpool.registration import (
    list_all_envs,
    make_dm,
    make_gym,
    make_gymnasium,
    make_spec,
)

_REPRESENTATIVE_ENVS = [
    "AdroitHandDoor-v1",
    "FetchReach-v4",
    "FetchReach-v1",
    "HandManipulatePen-v1",
    "HandManipulatePen-v0",
    "HandReach-v3",
    "HandReach-v0",
    "AntMaze_UMaze-v5",
    "PointMaze_UMaze-v3",
    "FrankaKitchen-v1",
]
_ALIASES = {
    "FetchReach-v1": "FetchReach-v4",
    "HandManipulatePen-v0": "HandManipulatePen-v1",
    "HandReach-v0": "HandReach-v3",
}

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _robotics_task_ids() -> list[str]:
    task_ids = sorted(
        task_id
        for task_id in list_all_envs()
        if task_id.startswith((
            "AdroitHand",
            "AntMaze_",
            "Fetch",
            "FrankaKitchen-",
            "HandManipulate",
            "HandReach",
            "PointMaze_",
        ))
    )
    assert len(task_ids) == 217, task_ids
    return task_ids


def _upstream_task_id(task_id: str) -> str:
    return robotics_registration._gymnasium_task_id(task_id)


def _make_upstream_env(task_id: str) -> gym.Env:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return gym.make(_upstream_task_id(task_id))


def _assert_space_equal(
    test_case: absltest.TestCase,
    lhs: gym.Space[Any],
    rhs: gym.Space[Any],
) -> None:
    test_case.assertIs(type(lhs), type(rhs))
    if isinstance(lhs, gym.spaces.Dict):
        rhs_dict = cast(gym.spaces.Dict, rhs)
        test_case.assertEqual(lhs.spaces.keys(), rhs_dict.spaces.keys())
        for key in lhs.spaces:
            _assert_space_equal(
                test_case,
                lhs.spaces[key],
                rhs_dict.spaces[key],
            )
        return
    if isinstance(lhs, gym.spaces.Box):
        rhs_box = cast(gym.spaces.Box, rhs)
        np.testing.assert_array_equal(lhs.low, rhs_box.low, err_msg="Box.low")
        np.testing.assert_array_equal(
            lhs.high,
            rhs_box.high,
            err_msg="Box.high",
        )
        test_case.assertEqual(lhs.shape, rhs_box.shape)
        test_case.assertEqual(lhs.dtype, rhs_box.dtype)
        return
    if isinstance(lhs, gym.spaces.Discrete):
        rhs_discrete = cast(gym.spaces.Discrete, rhs)
        test_case.assertEqual(lhs.n, rhs_discrete.n)
        test_case.assertEqual(lhs.start, rhs_discrete.start)
        test_case.assertEqual(lhs.dtype, rhs_discrete.dtype)
        return
    raise AssertionError(f"Unexpected space type: {lhs!r}")


def _assert_obs_equal(lhs: Any, rhs: Any) -> None:
    if isinstance(lhs, dict):
        assert isinstance(rhs, dict), rhs
        assert lhs.keys() == rhs.keys(), (lhs.keys(), rhs.keys())
        for key in lhs:
            _assert_obs_equal(lhs[key], rhs[key])
        return
    np.testing.assert_allclose(lhs, rhs, atol=1e-6, rtol=1e-6)


def _obs_allclose(lhs: Any, rhs: Any) -> bool:
    if isinstance(lhs, dict):
        return all(_obs_allclose(lhs[key], rhs[key]) for key in lhs)
    return bool(np.allclose(lhs, rhs, atol=1e-6, rtol=1e-6))


def _assert_info_value_equal(lhs: Any, rhs: Any) -> None:
    lhs_arr = np.asarray(lhs)
    rhs_arr = np.asarray(rhs)
    if np.issubdtype(lhs_arr.dtype, np.number) and np.issubdtype(
        rhs_arr.dtype, np.number
    ):
        np.testing.assert_allclose(lhs_arr, rhs_arr, atol=1e-6, rtol=1e-6)
    else:
        np.testing.assert_array_equal(lhs_arr, rhs_arr)


def _first_env_obs(obs: Any) -> Any:
    if isinstance(obs, dict):
        return {key: _first_env_obs(value) for key, value in obs.items()}
    return obs[0]


class _GymnasiumRoboticsEnvPoolTest(absltest.TestCase):
    def test_registered_robotics_env_count(self) -> None:
        self.assertLen(_robotics_task_ids(), 217)

    def test_make_registered_robotics_envs(self) -> None:
        for task_id in _robotics_task_ids():
            with self.subTest(task_id=task_id):
                spec = make_spec(task_id)
                self.assertIsNotNone(spec.observation_space)
                self.assertIsNotNone(spec.action_space)
                env_dm = make_dm(task_id)
                env_gym = make_gym(task_id)
                env_gymnasium = make_gymnasium(task_id)
                try:
                    self.assertIsInstance(env_dm, dm_env.Environment)
                    self.assertIsInstance(env_gym, gym.Env)
                    self.assertIsInstance(env_gymnasium, gym.Env)
                    env_dm.reset()
                    env_gym.reset()
                    env_gymnasium.reset()
                finally:
                    env_dm.close()
                    env_gym.close()
                    env_gymnasium.close()

    def test_space_alignment(self) -> None:
        for task_id in _robotics_task_ids():
            with self.subTest(task_id=task_id):
                env0 = _make_upstream_env(task_id)
                env1 = make_gymnasium(task_id)
                try:
                    _assert_space_equal(
                        self,
                        env0.observation_space,
                        env1.observation_space,
                    )
                    _assert_space_equal(
                        self,
                        env0.action_space,
                        env1.action_space,
                    )
                finally:
                    env0.close()
                    env1.close()

    def test_deterministic_rollout_same_seed(self) -> None:
        for task_id in _robotics_task_ids():
            with self.subTest(task_id=task_id):
                env0 = make_gymnasium(task_id, num_envs=2, seed=0)
                env1 = make_gymnasium(task_id, num_envs=2, seed=0)
                try:
                    _assert_obs_equal(env0.reset()[0], env1.reset()[0])
                    env0.action_space.seed(1)
                    for _ in range(8):
                        action = np.stack([
                            env0.action_space.sample() for _ in range(2)
                        ])
                        obs0, reward0, term0, trunc0, _ = env0.step(action)
                        obs1, reward1, term1, trunc1, _ = env1.step(action)
                        _assert_obs_equal(obs0, obs1)
                        np.testing.assert_allclose(reward0, reward1)
                        np.testing.assert_array_equal(term0, term1)
                        np.testing.assert_array_equal(trunc0, trunc1)
                finally:
                    env0.close()
                    env1.close()

    def test_different_seed_rollout_changes(self) -> None:
        for task_id in _REPRESENTATIVE_ENVS:
            with self.subTest(task_id=task_id):
                env0 = make_gymnasium(task_id, num_envs=2, seed=0)
                env1 = make_gymnasium(task_id, num_envs=2, seed=1)
                try:
                    obs0 = env0.reset()[0]
                    obs1 = env1.reset()[0]
                    differs = not _obs_allclose(obs0, obs1)
                    env0.action_space.seed(1)
                    for _ in range(8):
                        action = np.stack([
                            env0.action_space.sample() for _ in range(2)
                        ])
                        obs0 = env0.step(action)[0]
                        obs1 = env1.step(action)[0]
                        differs = differs or not _obs_allclose(obs0, obs1)
                        if differs:
                            break
                    self.assertTrue(
                        differs,
                        msg=f"expected different rollouts for {task_id}",
                    )
                finally:
                    env0.close()
                    env1.close()

    def test_align_with_upstream_rollout(self) -> None:
        for task_id in _robotics_task_ids():
            with self.subTest(task_id=task_id):
                env0 = _make_upstream_env(task_id)
                env1 = make_gymnasium(task_id, num_envs=1, seed=0)
                try:
                    obs0, info0 = env0.reset(seed=0)
                    obs1, info1 = env1.reset()
                    _assert_obs_equal(obs0, _first_env_obs(cast(Any, obs1)))
                    for key in set(info0) & set(info1):
                        _assert_info_value_equal(info0[key], info1[key][0])

                    env0.action_space.seed(3)
                    for _ in range(8):
                        action = env0.action_space.sample()
                        obs0, reward0, term0, trunc0, info0 = env0.step(action)
                        obs1, reward1, term1, trunc1, info1 = env1.step(
                            np.asarray([action]),
                            np.asarray([0], dtype=np.int32),
                        )
                        _assert_obs_equal(
                            obs0,
                            _first_env_obs(cast(Any, obs1)),
                        )
                        np.testing.assert_allclose(
                            np.asarray(reward0),
                            np.asarray(reward1[0]),
                        )
                        self.assertEqual(term0, term1[0])
                        self.assertEqual(trunc0, trunc1[0])
                        for key in set(info0) & set(info1):
                            _assert_info_value_equal(
                                info0[key],
                                info1[key][0],
                            )
                        if term0 or trunc0:
                            env0.reset()
                finally:
                    env0.close()
                    env1.close()

    def test_alias_ids_use_modern_robotics_tasks(self) -> None:
        for task_id, upstream_task_id in _ALIASES.items():
            with self.subTest(task_id=task_id):
                spec = make_spec(task_id)
                self.assertEqual(
                    spec.config.gymnasium_task_id,
                    upstream_task_id,
                )

    def test_dm_send_recv(self) -> None:
        env = make_dm("FetchReach-v4", num_envs=2, seed=0)
        try:
            timestep = env.reset()
            self.assertIsInstance(timestep, dm_env.TimeStep)
            self.assertEqual(timestep.reward.shape, (2,))

            env.action_space.seed(1)
            action = np.stack([env.action_space.sample() for _ in range(2)])
            env.send(action)
            timestep = env.recv()
            self.assertIsInstance(timestep, dm_env.TimeStep)
            self.assertEqual(timestep.reward.shape, (2,))
            self.assertEqual(timestep.discount.shape, (2,))
            self.assertEqual(timestep.step_type.shape, (2,))
        finally:
            env.close()

    def test_async_reset_recv(self) -> None:
        env_gymnasium = make_gymnasium("FetchReach-v4", num_envs=2, seed=0)
        env_dm = make_dm("FetchReach-v4", num_envs=2, seed=0)
        try:
            env_gymnasium.async_reset()
            obs, reward, terminated, truncated, info = env_gymnasium.recv()
            self.assertIsInstance(obs, dict)
            self.assertEqual(obs["observation"].shape[0], 2)
            self.assertEqual(reward.tolist(), [0.0, 0.0])
            self.assertEqual(terminated.tolist(), [False, False])
            self.assertEqual(truncated.tolist(), [False, False])
            self.assertEqual(info["env_id"].tolist(), [0, 1])
            self.assertEqual(info["elapsed_step"].tolist(), [0, 0])

            env_dm.async_reset()
            timestep = env_dm.recv()
            self.assertIsInstance(timestep, dm_env.TimeStep)
            self.assertTrue(timestep.first().all())
            self.assertEqual(timestep.reward.tolist(), [0.0, 0.0])
            self.assertEqual(timestep.discount.tolist(), [1.0, 1.0])
        finally:
            env_gymnasium.close()
            env_dm.close()

    def test_render_smoke(self) -> None:
        env = make_gymnasium(
            "FetchReach-v4",
            render_mode="rgb_array",
        )
        try:
            env.reset()
            try:
                frame = env.render()
            except Exception as error:
                if sys.platform in {"darwin", "win32"} and "gladLoadGL" in str(
                    error
                ):
                    self.skipTest(
                        "Gymnasium-Robotics MuJoCo offscreen rendering is "
                        "unavailable in this runtime."
                    )
                raise
            self.assertIsNotNone(frame)
            frame = cast(np.ndarray, frame)
            self.assertEqual(frame.shape[0], 1)
            self.assertEqual(frame.shape[-1], 3)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
