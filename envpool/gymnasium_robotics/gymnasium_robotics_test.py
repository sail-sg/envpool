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
"""Tests for native C++ Gymnasium-Robotics envs."""

from __future__ import annotations

import sys
import warnings
from typing import Any, cast

import dm_env
import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import mujoco
import numpy as np
from absl.testing import absltest

import envpool.gymnasium_robotics.registration as robotics_registration
from envpool.registration import (
    list_all_envs,
    make_dm,
    make_gymnasium,
    make_spec,
)

_FETCH_ENVS = robotics_registration.gymnasium_robotics_fetch_envs
_HAND_ENVS = robotics_registration.gymnasium_robotics_hand_envs
_ADROIT_ENVS = robotics_registration.gymnasium_robotics_adroit_envs
_POINT_MAZE_ENVS = robotics_registration.gymnasium_robotics_point_maze_envs
_KITCHEN_ENVS = robotics_registration.gymnasium_robotics_kitchen_envs
_FETCH_V4_BY_V1 = {
    "FetchPickAndPlace-v1": "FetchPickAndPlace-v4",
    "FetchPickAndPlaceDense-v1": "FetchPickAndPlaceDense-v4",
    "FetchPush-v1": "FetchPush-v4",
    "FetchPushDense-v1": "FetchPushDense-v4",
    "FetchReach-v1": "FetchReach-v4",
    "FetchReachDense-v1": "FetchReachDense-v4",
    "FetchSlide-v1": "FetchSlide-v4",
    "FetchSlideDense-v1": "FetchSlideDense-v4",
}
_HAND_CANONICAL_BY_V0 = {
    task_id: (
        task_id.replace("-v0", "-v3")
        if task_id.startswith("HandReach")
        else task_id.replace("-v0", "-v1")
    )
    for task_id in _HAND_ENVS
    if task_id.endswith("-v0")
}

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _upstream_task_id(task_id: str) -> str:
    if task_id in _FETCH_V4_BY_V1:
        return _FETCH_V4_BY_V1[task_id]
    return _HAND_CANONICAL_BY_V0.get(task_id, task_id)


def _make_upstream_env(task_id: str, **kwargs: Any) -> gym.Env:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return gym.make(_upstream_task_id(task_id), **kwargs)


def _assert_goal_obs_equal(
    lhs: Any,
    rhs: Any,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    if isinstance(lhs, dict):
        assert isinstance(rhs, dict), rhs
        assert lhs.keys() == rhs.keys(), (lhs.keys(), rhs.keys())
        for key in lhs:
            _assert_goal_obs_equal(lhs[key], rhs[key], atol=atol, rtol=rtol)
        return
    np.testing.assert_allclose(lhs, rhs, atol=atol, rtol=rtol)


def _assert_space_equal(lhs: gym.Space, rhs: gym.Space) -> None:
    assert type(lhs) is type(rhs), (type(lhs), type(rhs))
    if isinstance(lhs, gym.spaces.Dict):
        assert isinstance(rhs, gym.spaces.Dict)
        assert lhs.spaces.keys() == rhs.spaces.keys(), (
            lhs.spaces.keys(),
            rhs.spaces.keys(),
        )
        for key in lhs.spaces:
            _assert_space_equal(lhs.spaces[key], rhs.spaces[key])
        return
    assert isinstance(lhs, gym.spaces.Box)
    assert isinstance(rhs, gym.spaces.Box)
    np.testing.assert_allclose(lhs.low, rhs.low)
    np.testing.assert_allclose(lhs.high, rhs.high)
    assert lhs.dtype == rhs.dtype, (lhs.dtype, rhs.dtype)


def _assert_scalar_allclose(
    lhs: Any,
    rhs: Any,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    np.testing.assert_allclose(
        np.asarray(lhs),
        np.asarray(rhs),
        atol=atol,
        rtol=rtol,
    )


def _obs_allclose(lhs: Any, rhs: Any) -> bool:
    if isinstance(lhs, dict):
        return lhs.keys() == rhs.keys() and all(
            _obs_allclose(lhs[key], rhs[key]) for key in lhs
        )
    return np.allclose(lhs, rhs, atol=1e-6, rtol=1e-6)


def _first_env_obs(obs: Any) -> Any:
    if isinstance(obs, dict):
        return {key: _first_env_obs(value) for key, value in obs.items()}
    return obs[0]


def _reset_upstream_state(
    env: gym.Env,
    qpos: np.ndarray,
    qvel: np.ndarray,
    goal: np.ndarray,
) -> dict[str, np.ndarray]:
    base_env = cast(Any, env.unwrapped)
    mujoco.mj_resetData(base_env.model, base_env.data)
    base_env.data.qpos[:] = qpos
    base_env.data.qvel[:] = qvel
    if hasattr(base_env, "_set_action"):
        base_env._set_action(
            np.zeros(base_env.action_space.shape, dtype=base_env.action_space.dtype)
        )
    mujoco.mj_forward(base_env.model, base_env.data)
    base_env.goal = np.array(goal, copy=True)
    return base_env._get_obs()


def _reset_upstream_adroit_state(
    env: gym.Env,
    qpos: np.ndarray,
    qvel: np.ndarray,
    extra: np.ndarray,
    task_id: str,
) -> np.ndarray:
    base_env = cast(Any, env.unwrapped)
    if task_id.startswith("AdroitHandDoor"):
        body_id = mujoco.mj_name2id(base_env.model, mujoco.mjtObj.mjOBJ_BODY, "frame")
        base_env.model.body_pos[body_id] = extra
    elif task_id.startswith("AdroitHandHammer"):
        body_id = mujoco.mj_name2id(
            base_env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "nail_board",
        )
        base_env.model.body_pos[body_id] = extra
    elif task_id.startswith("AdroitHandPen"):
        body_id = mujoco.mj_name2id(
            base_env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target",
        )
        base_env.model.body_quat[body_id] = extra
    elif task_id.startswith("AdroitHandRelocate"):
        body_id = mujoco.mj_name2id(
            base_env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "Object",
        )
        site_id = mujoco.mj_name2id(
            base_env.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "target",
        )
        base_env.model.body_pos[body_id] = extra[:3]
        base_env.model.site_pos[site_id] = extra[3:]
    else:
        raise ValueError(f"Unsupported Adroit task: {task_id}")

    mujoco.mj_resetData(base_env.model, base_env.data)
    base_env.data.qpos[:] = qpos
    base_env.data.qvel[:] = qvel
    mujoco.mj_forward(base_env.model, base_env.data)
    if task_id.startswith("AdroitHandPen"):
        base_env.pen_length = np.linalg.norm(
            base_env.data.site_xpos[base_env.obj_t_site_id]
            - base_env.data.site_xpos[base_env.obj_b_site_id]
        )
        base_env.tar_length = np.linalg.norm(
            base_env.data.site_xpos[base_env.tar_t_site_id]
            - base_env.data.site_xpos[base_env.tar_b_site_id]
        )
    return cast(np.ndarray, base_env._get_obs())


def _reset_upstream_point_maze_state(
    env: gym.Env,
    qpos: np.ndarray,
    qvel: np.ndarray,
    goal: np.ndarray,
) -> dict[str, np.ndarray]:
    base_env = cast(Any, env.unwrapped)
    mujoco.mj_resetData(base_env.model, base_env.data)
    base_env.data.qpos[:] = qpos
    base_env.data.qvel[:] = qvel
    base_env.goal = np.array(goal, copy=True)
    base_env.update_target_site_pos()
    mujoco.mj_forward(base_env.model, base_env.data)
    point_obs, _ = base_env.point_env._get_obs()
    return base_env._get_obs(point_obs)


def _reset_upstream_kitchen_state(
    env: gym.Env,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> dict[str, Any]:
    base_env = cast(Any, env.unwrapped)
    mujoco.mj_resetData(base_env.robot_env.model, base_env.robot_env.data)
    base_env.robot_env.data.qpos[:] = qpos
    base_env.robot_env.data.qvel[:] = qvel
    mujoco.mj_forward(base_env.robot_env.model, base_env.robot_env.data)
    base_env.tasks_to_complete = set(base_env.goal.keys())
    base_env.step_task_completions = []
    base_env.episode_task_completions = []
    robot_obs = base_env.robot_env._get_obs()
    return base_env._get_obs(robot_obs)


def _kitchen_task_list_to_mask(tasks: list[str]) -> np.ndarray:
    mask = np.zeros(7, dtype=np.int32)
    for task_id, task_name in enumerate(
        [
            "bottom burner",
            "top burner",
            "light switch",
            "slide cabinet",
            "hinge cabinet",
            "microwave",
            "kettle",
        ]
    ):
        if task_name in tasks:
            mask[task_id] = 1
    return mask


class _GymnasiumRoboticsFetchEnvPoolTest(absltest.TestCase):
    def test_registered_fetch_env_count(self) -> None:
        task_ids = sorted(
            task_id for task_id in list_all_envs() if task_id.startswith("Fetch")
        )
        self.assertEqual(task_ids, sorted(_FETCH_ENVS))
        self.assertLen(task_ids, 16)

    def test_make_registered_fetch_envs(self) -> None:
        for task_id in _FETCH_ENVS:
            with self.subTest(task_id=task_id):
                spec = make_spec(task_id)
                self.assertIsNotNone(spec.observation_space)
                self.assertIsNotNone(spec.action_space)

                env_dm = make_dm(task_id)
                env_gymnasium = make_gymnasium(task_id)
                try:
                    self.assertIsInstance(env_dm, dm_env.Environment)
                    self.assertIsInstance(env_gymnasium, gym.Env)
                    env_dm.reset()
                    env_gymnasium.reset()
                finally:
                    env_dm.close()
                    env_gymnasium.close()

    def test_space_alignment(self) -> None:
        for task_id in _FETCH_ENVS:
            with self.subTest(task_id=task_id):
                env0 = _make_upstream_env(task_id)
                env1 = make_gymnasium(task_id)
                try:
                    _assert_space_equal(
                        env0.observation_space,
                        env1.observation_space,
                    )
                    _assert_space_equal(env0.action_space, env1.action_space)
                finally:
                    env0.close()
                    env1.close()

    def test_deterministic_rollout_same_seed(self) -> None:
        for task_id in _FETCH_ENVS:
            with self.subTest(task_id=task_id):
                env0 = make_gymnasium(task_id, num_envs=2, seed=0)
                env1 = make_gymnasium(task_id, num_envs=2, seed=0)
                try:
                    _assert_goal_obs_equal(env0.reset()[0], env1.reset()[0])
                    env0.action_space.seed(1)
                    for _ in range(32):
                        action = np.stack(
                            [env0.action_space.sample() for _ in range(2)]
                        )
                        obs0, reward0, term0, trunc0, _ = env0.step(action)
                        obs1, reward1, term1, trunc1, _ = env1.step(action)
                        _assert_goal_obs_equal(obs0, obs1)
                        np.testing.assert_allclose(reward0, reward1)
                        np.testing.assert_array_equal(term0, term1)
                        np.testing.assert_array_equal(trunc0, trunc1)
                finally:
                    env0.close()
                    env1.close()

    def test_different_seed_rollout_changes(self) -> None:
        for task_id in _FETCH_ENVS:
            with self.subTest(task_id=task_id):
                env0 = make_gymnasium(task_id, num_envs=2, seed=0)
                env1 = make_gymnasium(task_id, num_envs=2, seed=1)
                try:
                    obs0 = env0.reset()[0]
                    obs1 = env1.reset()[0]
                    differs = not _obs_allclose(obs0, obs1)
                    env0.action_space.seed(1)
                    for _ in range(32):
                        action = np.stack(
                            [env0.action_space.sample() for _ in range(2)]
                        )
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
        for task_id in _FETCH_ENVS:
            with self.subTest(task_id=task_id):
                env0 = _make_upstream_env(task_id)
                env1 = make_gymnasium(task_id, num_envs=1, seed=0)
                try:
                    env0.action_space.seed(3)
                    env0.reset(seed=0)
                    obs1, info1 = env1.reset()
                    obs0 = _reset_upstream_state(
                        env0,
                        info1["qpos0"][0],
                        info1["qvel0"][0],
                        info1["goal0"][0],
                    )
                    _assert_goal_obs_equal(
                        obs0,
                        _first_env_obs(cast(Any, obs1)),
                        atol=1e-4,
                        rtol=1e-4,
                    )

                    terminated1 = np.array([False])
                    truncated1 = np.array([False])
                    for _ in range(32):
                        action = env0.action_space.sample()
                        obs0, reward0, terminated0, truncated0, info0 = env0.step(
                            action
                        )
                        obs1, reward1, terminated1, truncated1, info1 = env1.step(
                            np.asarray([action], dtype=env0.action_space.dtype),
                            np.asarray([0], dtype=np.int32),
                        )
                        _assert_goal_obs_equal(
                            obs0,
                            _first_env_obs(cast(Any, obs1)),
                            atol=1e-4,
                            rtol=1e-4,
                        )
                        _assert_scalar_allclose(
                            reward0,
                            reward1[0],
                            atol=1e-6,
                            rtol=1e-6,
                        )
                        self.assertEqual(terminated0, terminated1[0])
                        self.assertEqual(truncated0, truncated1[0])
                        np.testing.assert_allclose(
                            info0["is_success"],
                            info1["is_success"][0],
                        )
                        np.testing.assert_allclose(
                            np.linalg.norm(
                                obs0["achieved_goal"] - obs0["desired_goal"]
                            ),
                            info1["distance"][0],
                            atol=1e-6,
                            rtol=1e-6,
                        )
                        if terminated1[0] or truncated1[0]:
                            break
                finally:
                    env0.close()
                    env1.close()

    def test_v1_alias_matches_v4_spec(self) -> None:
        for alias_id, target_id in _FETCH_V4_BY_V1.items():
            with self.subTest(alias_id=alias_id):
                alias_spec = make_spec(alias_id)
                target_spec = make_spec(target_id)
                self.assertEqual(alias_spec.config.xml_file, target_spec.config.xml_file)
                self.assertEqual(
                    alias_spec.config.reward_type,
                    target_spec.config.reward_type,
                )
                self.assertEqual(
                    alias_spec.observation_space,
                    target_spec.observation_space,
                )
                self.assertEqual(alias_spec.action_space, target_spec.action_space)

    def test_dm_send_recv(self) -> None:
        action_space = make_spec("FetchReach-v4").action_space
        env = make_dm("FetchReach-v4", num_envs=2, seed=0)
        try:
            timestep = env.reset()
            self.assertIsInstance(timestep, dm_env.TimeStep)
            self.assertEqual(timestep.reward.shape, (2,))

            action_space.seed(1)
            action = np.stack([action_space.sample() for _ in range(2)])
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
        env = make_gymnasium("FetchReach-v4", render_mode="rgb_array")
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


class _GymnasiumRoboticsHandEnvPoolTest(absltest.TestCase):
    def test_registered_hand_env_count(self) -> None:
        task_ids = sorted(
            task_id for task_id in list_all_envs() if task_id.startswith("Hand")
        )
        self.assertEqual(task_ids, sorted(_HAND_ENVS))
        self.assertLen(task_ids, 112)

    def test_make_registered_hand_envs(self) -> None:
        for task_id in _HAND_ENVS:
            with self.subTest(task_id=task_id):
                spec = make_spec(task_id)
                self.assertIsNotNone(spec.observation_space)
                self.assertIsNotNone(spec.action_space)

                env_dm = make_dm(task_id)
                env_gymnasium = make_gymnasium(task_id)
                try:
                    self.assertIsInstance(env_dm, dm_env.Environment)
                    self.assertIsInstance(env_gymnasium, gym.Env)
                    env_dm.reset()
                    env_gymnasium.reset()
                finally:
                    env_dm.close()
                    env_gymnasium.close()

    def test_space_alignment(self) -> None:
        for task_id in _HAND_ENVS:
            with self.subTest(task_id=task_id):
                env0 = _make_upstream_env(task_id)
                env1 = make_gymnasium(task_id)
                try:
                    _assert_space_equal(
                        env0.observation_space,
                        env1.observation_space,
                    )
                    _assert_space_equal(env0.action_space, env1.action_space)
                finally:
                    env0.close()
                    env1.close()

    def test_deterministic_rollout_same_seed(self) -> None:
        for task_id in _HAND_ENVS:
            with self.subTest(task_id=task_id):
                env0 = make_gymnasium(task_id, num_envs=2, seed=0)
                env1 = make_gymnasium(task_id, num_envs=2, seed=0)
                try:
                    _assert_goal_obs_equal(env0.reset()[0], env1.reset()[0])
                    env0.action_space.seed(1)
                    for _ in range(8):
                        action = np.stack(
                            [env0.action_space.sample() for _ in range(2)]
                        )
                        obs0, reward0, term0, trunc0, _ = env0.step(action)
                        obs1, reward1, term1, trunc1, _ = env1.step(action)
                        _assert_goal_obs_equal(obs0, obs1)
                        np.testing.assert_allclose(reward0, reward1)
                        np.testing.assert_array_equal(term0, term1)
                        np.testing.assert_array_equal(trunc0, trunc1)
                finally:
                    env0.close()
                    env1.close()

    def test_different_seed_rollout_changes(self) -> None:
        for task_id in _HAND_ENVS:
            with self.subTest(task_id=task_id):
                env0 = make_gymnasium(task_id, num_envs=2, seed=0)
                env1 = make_gymnasium(task_id, num_envs=2, seed=1)
                try:
                    obs0 = env0.reset()[0]
                    obs1 = env1.reset()[0]
                    differs = not _obs_allclose(obs0, obs1)
                    env0.action_space.seed(1)
                    for _ in range(8):
                        action = np.stack(
                            [env0.action_space.sample() for _ in range(2)]
                        )
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
        for task_id in _HAND_ENVS:
            with self.subTest(task_id=task_id):
                env0 = _make_upstream_env(task_id)
                env1 = make_gymnasium(task_id, num_envs=1, seed=0)
                try:
                    env0.action_space.seed(3)
                    env0.reset(seed=0)
                    obs1, info1 = env1.reset()
                    obs0 = _reset_upstream_state(
                        env0,
                        info1["qpos0"][0],
                        info1["qvel0"][0],
                        info1["goal0"][0],
                    )
                    _assert_goal_obs_equal(
                        obs0,
                        _first_env_obs(cast(Any, obs1)),
                        atol=1e-4,
                        rtol=1e-4,
                    )

                    terminated1 = np.array([False])
                    truncated1 = np.array([False])
                    for _ in range(8):
                        action = env0.action_space.sample()
                        obs0, reward0, terminated0, truncated0, info0 = env0.step(
                            action
                        )
                        obs1, reward1, terminated1, truncated1, info1 = env1.step(
                            np.asarray([action], dtype=env0.action_space.dtype),
                            np.asarray([0], dtype=np.int32),
                        )
                        _assert_goal_obs_equal(
                            obs0,
                            _first_env_obs(cast(Any, obs1)),
                            atol=1e-4,
                            rtol=1e-4,
                        )
                        _assert_scalar_allclose(
                            reward0,
                            reward1[0],
                            atol=1e-5,
                            rtol=1e-5,
                        )
                        self.assertEqual(terminated0, terminated1[0])
                        self.assertEqual(truncated0, truncated1[0])
                        np.testing.assert_allclose(
                            info0["is_success"],
                            info1["is_success"][0],
                        )
                        if terminated1[0] or truncated1[0]:
                            break
                finally:
                    env0.close()
                    env1.close()

    def test_v0_alias_matches_canonical_spec(self) -> None:
        for alias_id, target_id in _HAND_CANONICAL_BY_V0.items():
            with self.subTest(alias_id=alias_id):
                alias_spec = make_spec(alias_id)
                target_spec = make_spec(target_id)
                self.assertEqual(alias_spec.config.xml_file, target_spec.config.xml_file)
                self.assertEqual(
                    alias_spec.config.reward_type,
                    target_spec.config.reward_type,
                )
                self.assertEqual(
                    alias_spec.observation_space,
                    target_spec.observation_space,
                )
                self.assertEqual(alias_spec.action_space, target_spec.action_space)


class _GymnasiumRoboticsAdroitEnvPoolTest(absltest.TestCase):
    def test_registered_adroit_env_count(self) -> None:
        task_ids = sorted(
            task_id
            for task_id in list_all_envs()
            if task_id.startswith("AdroitHand")
        )
        self.assertEqual(task_ids, sorted(_ADROIT_ENVS))
        self.assertLen(task_ids, 8)

    def test_make_registered_adroit_envs(self) -> None:
        for task_id in _ADROIT_ENVS:
            with self.subTest(task_id=task_id):
                spec = make_spec(task_id)
                self.assertIsNotNone(spec.observation_space)
                self.assertIsNotNone(spec.action_space)

                env_dm = make_dm(task_id)
                env_gymnasium = make_gymnasium(task_id)
                try:
                    self.assertIsInstance(env_dm, dm_env.Environment)
                    self.assertIsInstance(env_gymnasium, gym.Env)
                    env_dm.reset()
                    env_gymnasium.reset()
                finally:
                    env_dm.close()
                    env_gymnasium.close()

    def test_space_alignment(self) -> None:
        for task_id in _ADROIT_ENVS:
            with self.subTest(task_id=task_id):
                env0 = _make_upstream_env(task_id)
                env1 = make_gymnasium(task_id)
                try:
                    _assert_space_equal(env0.observation_space, env1.observation_space)
                    _assert_space_equal(env0.action_space, env1.action_space)
                finally:
                    env0.close()
                    env1.close()

    def test_deterministic_rollout_same_seed(self) -> None:
        for task_id in _ADROIT_ENVS:
            with self.subTest(task_id=task_id):
                env0 = make_gymnasium(task_id, num_envs=2, seed=0)
                env1 = make_gymnasium(task_id, num_envs=2, seed=0)
                try:
                    np.testing.assert_allclose(env0.reset()[0], env1.reset()[0])
                    env0.action_space.seed(1)
                    for _ in range(8):
                        action = np.stack(
                            [env0.action_space.sample() for _ in range(2)]
                        )
                        obs0, reward0, term0, trunc0, _ = env0.step(action)
                        obs1, reward1, term1, trunc1, _ = env1.step(action)
                        np.testing.assert_allclose(obs0, obs1)
                        np.testing.assert_allclose(reward0, reward1)
                        np.testing.assert_array_equal(term0, term1)
                        np.testing.assert_array_equal(trunc0, trunc1)
                finally:
                    env0.close()
                    env1.close()

    def test_different_seed_rollout_changes(self) -> None:
        for task_id in _ADROIT_ENVS:
            with self.subTest(task_id=task_id):
                env0 = make_gymnasium(task_id, num_envs=2, seed=0)
                env1 = make_gymnasium(task_id, num_envs=2, seed=1)
                try:
                    obs0 = env0.reset()[0]
                    obs1 = env1.reset()[0]
                    differs = not np.allclose(obs0, obs1)
                    env0.action_space.seed(1)
                    for _ in range(8):
                        action = np.stack(
                            [env0.action_space.sample() for _ in range(2)]
                        )
                        obs0 = env0.step(action)[0]
                        obs1 = env1.step(action)[0]
                        differs = differs or not np.allclose(obs0, obs1)
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
        for task_id in _ADROIT_ENVS:
            with self.subTest(task_id=task_id):
                env0 = _make_upstream_env(task_id)
                env1 = make_gymnasium(task_id, num_envs=1, seed=0)
                try:
                    env0.action_space.seed(3)
                    env0.reset(seed=0)
                    obs1, info1 = env1.reset()
                    obs0 = _reset_upstream_adroit_state(
                        env0,
                        info1["qpos0"][0],
                        info1["qvel0"][0],
                        info1["extra0"][0],
                        task_id,
                    )
                    np.testing.assert_allclose(
                        obs0,
                        obs1[0],
                        atol=1e-4,
                        rtol=1e-4,
                    )

                    for _ in range(8):
                        action = env0.action_space.sample()
                        obs0, reward0, terminated0, truncated0, info0 = env0.step(
                            action
                        )
                        obs1, reward1, terminated1, truncated1, info1 = env1.step(
                            np.asarray([action], dtype=env0.action_space.dtype),
                            np.asarray([0], dtype=np.int32),
                        )
                        np.testing.assert_allclose(
                            obs0,
                            obs1[0],
                            atol=1e-4,
                            rtol=1e-4,
                        )
                        _assert_scalar_allclose(
                            reward0,
                            reward1[0],
                            atol=1e-5,
                            rtol=1e-5,
                        )
                        self.assertEqual(terminated0, terminated1[0])
                        self.assertEqual(truncated0, truncated1[0])
                        np.testing.assert_allclose(
                            info0["success"],
                            info1["success"][0],
                        )
                        if terminated1[0] or truncated1[0]:
                            break
                finally:
                    env0.close()
                    env1.close()


class _GymnasiumRoboticsPointMazeEnvPoolTest(absltest.TestCase):
    def test_registered_point_maze_env_count(self) -> None:
        task_ids = sorted(
            task_id
            for task_id in list_all_envs()
            if task_id.startswith("PointMaze_")
        )
        self.assertEqual(task_ids, sorted(_POINT_MAZE_ENVS))
        self.assertLen(task_ids, 20)

    def test_make_registered_point_maze_envs(self) -> None:
        for task_id in _POINT_MAZE_ENVS:
            with self.subTest(task_id=task_id):
                spec = make_spec(task_id)
                self.assertIsNotNone(spec.observation_space)
                self.assertIsNotNone(spec.action_space)

                env_dm = make_dm(task_id)
                env_gymnasium = make_gymnasium(task_id)
                try:
                    self.assertIsInstance(env_dm, dm_env.Environment)
                    self.assertIsInstance(env_gymnasium, gym.Env)
                    env_dm.reset()
                    env_gymnasium.reset()
                finally:
                    env_dm.close()
                    env_gymnasium.close()

    def test_space_alignment(self) -> None:
        for task_id in _POINT_MAZE_ENVS:
            with self.subTest(task_id=task_id):
                env0 = _make_upstream_env(task_id)
                env1 = make_gymnasium(task_id)
                try:
                    _assert_space_equal(env0.observation_space, env1.observation_space)
                    _assert_space_equal(env0.action_space, env1.action_space)
                finally:
                    env0.close()
                    env1.close()

    def test_deterministic_rollout_same_seed(self) -> None:
        for task_id in _POINT_MAZE_ENVS:
            with self.subTest(task_id=task_id):
                env0 = make_gymnasium(task_id, num_envs=2, seed=0)
                env1 = make_gymnasium(task_id, num_envs=2, seed=0)
                try:
                    _assert_goal_obs_equal(env0.reset()[0], env1.reset()[0])
                    env0.action_space.seed(1)
                    for _ in range(16):
                        action = np.stack(
                            [env0.action_space.sample() for _ in range(2)]
                        )
                        obs0, reward0, term0, trunc0, _ = env0.step(action)
                        obs1, reward1, term1, trunc1, _ = env1.step(action)
                        _assert_goal_obs_equal(obs0, obs1)
                        np.testing.assert_allclose(reward0, reward1)
                        np.testing.assert_array_equal(term0, term1)
                        np.testing.assert_array_equal(trunc0, trunc1)
                finally:
                    env0.close()
                    env1.close()

    def test_different_seed_rollout_changes(self) -> None:
        for task_id in _POINT_MAZE_ENVS:
            with self.subTest(task_id=task_id):
                env0 = make_gymnasium(task_id, num_envs=2, seed=0)
                env1 = make_gymnasium(task_id, num_envs=2, seed=1)
                try:
                    obs0 = env0.reset()[0]
                    obs1 = env1.reset()[0]
                    differs = not _obs_allclose(obs0, obs1)
                    env0.action_space.seed(1)
                    for _ in range(16):
                        action = np.stack(
                            [env0.action_space.sample() for _ in range(2)]
                        )
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
        for task_id in _POINT_MAZE_ENVS:
            with self.subTest(task_id=task_id):
                env0 = _make_upstream_env(task_id)
                env1 = make_gymnasium(task_id, num_envs=1, seed=0)
                try:
                    env0.action_space.seed(3)
                    env0.reset(seed=0)
                    obs1, info1 = env1.reset()
                    obs0 = _reset_upstream_point_maze_state(
                        env0,
                        info1["qpos0"][0],
                        info1["qvel0"][0],
                        info1["goal0"][0],
                    )
                    _assert_goal_obs_equal(
                        obs0,
                        _first_env_obs(cast(Any, obs1)),
                        atol=1e-4,
                        rtol=1e-4,
                    )

                    for _ in range(16):
                        action = env0.action_space.sample()
                        obs0, reward0, terminated0, truncated0, info0 = env0.step(
                            action
                        )
                        obs1, reward1, terminated1, truncated1, info1 = env1.step(
                            np.asarray([action], dtype=env0.action_space.dtype),
                            np.asarray([0], dtype=np.int32),
                        )
                        _assert_goal_obs_equal(
                            obs0,
                            _first_env_obs(cast(Any, obs1)),
                            atol=1e-4,
                            rtol=1e-4,
                        )
                        _assert_scalar_allclose(
                            reward0,
                            reward1[0],
                            atol=1e-5,
                            rtol=1e-5,
                        )
                        self.assertEqual(terminated0, terminated1[0])
                        self.assertEqual(truncated0, truncated1[0])
                        np.testing.assert_allclose(
                            info0["success"],
                            info1["success"][0],
                        )
                        if terminated1[0] or truncated1[0]:
                            break
                finally:
                    env0.close()
                    env1.close()


class _GymnasiumRoboticsKitchenEnvPoolTest(absltest.TestCase):
    def test_registered_kitchen_env_count(self) -> None:
        task_ids = sorted(
            task_id
            for task_id in list_all_envs()
            if task_id.startswith("FrankaKitchen")
        )
        self.assertEqual(task_ids, sorted(_KITCHEN_ENVS))
        self.assertLen(task_ids, 1)

    def test_make_registered_kitchen_envs(self) -> None:
        for task_id in _KITCHEN_ENVS:
            with self.subTest(task_id=task_id):
                spec = make_spec(task_id)
                self.assertIsNotNone(spec.observation_space)
                self.assertIsNotNone(spec.action_space)

                env_dm = make_dm(task_id)
                env_gymnasium = make_gymnasium(task_id)
                try:
                    self.assertIsInstance(env_dm, dm_env.Environment)
                    self.assertIsInstance(env_gymnasium, gym.Env)
                    env_dm.reset()
                    env_gymnasium.reset()
                finally:
                    env_dm.close()
                    env_gymnasium.close()

    def test_space_alignment(self) -> None:
        env0 = _make_upstream_env("FrankaKitchen-v1")
        env1 = make_gymnasium("FrankaKitchen-v1")
        try:
            _assert_space_equal(env0.observation_space, env1.observation_space)
            _assert_space_equal(env0.action_space, env1.action_space)
        finally:
            env0.close()
            env1.close()

    def test_deterministic_rollout_same_seed(self) -> None:
        env0 = make_gymnasium("FrankaKitchen-v1", num_envs=2, seed=0)
        env1 = make_gymnasium("FrankaKitchen-v1", num_envs=2, seed=0)
        try:
            _assert_goal_obs_equal(env0.reset()[0], env1.reset()[0])
            env0.action_space.seed(1)
            for _ in range(8):
                action = np.stack([env0.action_space.sample() for _ in range(2)])
                obs0, reward0, term0, trunc0, _ = env0.step(action)
                obs1, reward1, term1, trunc1, _ = env1.step(action)
                _assert_goal_obs_equal(obs0, obs1)
                np.testing.assert_allclose(reward0, reward1)
                np.testing.assert_array_equal(term0, term1)
                np.testing.assert_array_equal(trunc0, trunc1)
        finally:
            env0.close()
            env1.close()

    def test_different_seed_rollout_changes(self) -> None:
        env0 = make_gymnasium("FrankaKitchen-v1", num_envs=2, seed=0)
        env1 = make_gymnasium("FrankaKitchen-v1", num_envs=2, seed=1)
        try:
            obs0 = env0.reset()[0]
            obs1 = env1.reset()[0]
            differs = not _obs_allclose(obs0, obs1)
            env0.action_space.seed(1)
            for _ in range(8):
                action = np.stack([env0.action_space.sample() for _ in range(2)])
                obs0 = env0.step(action)[0]
                obs1 = env1.step(action)[0]
                differs = differs or not _obs_allclose(obs0, obs1)
                if differs:
                    break
            self.assertTrue(differs, msg="expected different kitchen rollouts")
        finally:
            env0.close()
            env1.close()

    def test_align_with_upstream_rollout(self) -> None:
        env0 = _make_upstream_env(
            "FrankaKitchen-v1",
            robot_noise_ratio=0.0,
            object_noise_ratio=0.0,
        )
        env1 = make_gymnasium(
            "FrankaKitchen-v1",
            num_envs=1,
            seed=0,
            robot_noise_ratio=0.0,
            object_noise_ratio=0.0,
        )
        try:
            env0.action_space.seed(3)
            env0.reset(seed=0)
            obs1, info1 = env1.reset()
            obs0 = _reset_upstream_kitchen_state(
                env0,
                info1["qpos0"][0],
                info1["qvel0"][0],
            )
            _assert_goal_obs_equal(
                obs0,
                _first_env_obs(cast(Any, obs1)),
                atol=1e-4,
                rtol=1e-4,
            )
            np.testing.assert_array_equal(
                info1["tasks_to_complete"][0],
                np.ones(7, dtype=np.int32),
            )
            np.testing.assert_array_equal(
                info1["step_task_completions"][0],
                np.zeros(7, dtype=np.int32),
            )
            np.testing.assert_array_equal(
                info1["episode_task_completions"][0],
                np.zeros(7, dtype=np.int32),
            )

            for _ in range(8):
                action = env0.action_space.sample()
                obs0, reward0, terminated0, truncated0, info0 = env0.step(action)
                obs1, reward1, terminated1, truncated1, info1 = env1.step(
                    np.asarray([action], dtype=env0.action_space.dtype),
                    np.asarray([0], dtype=np.int32),
                )
                _assert_goal_obs_equal(
                    obs0,
                    _first_env_obs(cast(Any, obs1)),
                    atol=1e-4,
                    rtol=1e-4,
                )
                _assert_scalar_allclose(
                    reward0,
                    reward1[0],
                    atol=1e-5,
                    rtol=1e-5,
                )
                self.assertEqual(terminated0, terminated1[0])
                self.assertEqual(truncated0, truncated1[0])
                np.testing.assert_array_equal(
                    _kitchen_task_list_to_mask(info0["tasks_to_complete"]),
                    info1["tasks_to_complete"][0],
                )
                np.testing.assert_array_equal(
                    _kitchen_task_list_to_mask(info0["step_task_completions"]),
                    info1["step_task_completions"][0],
                )
                np.testing.assert_array_equal(
                    _kitchen_task_list_to_mask(
                        info0["episode_task_completions"]
                    ),
                    info1["episode_task_completions"][0],
                )
                if terminated1[0] or truncated1[0]:
                    break
        finally:
            env0.close()
            env1.close()


if __name__ == "__main__":
    absltest.main()
