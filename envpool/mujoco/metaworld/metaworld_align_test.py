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
"""Alignment tests against the official MetaWorld v3.0.0 oracle."""

from __future__ import annotations

import platform
import sys
from typing import Any

import mujoco
import numpy as np
from absl.testing import absltest
from metaworld.env_dict import ALL_V3_ENVIRONMENTS

import envpool.mujoco.metaworld.registration as metaworld_registration
from envpool.registration import make_gymnasium

_ALIGN_STEPS = 128
# The native C++ path and the official Python oracle both step MuJoCo, but they
# cross the C++/Python boundary differently. After reset-time state sync the
# residual over the 128-step rollout is still sub-1e-8 on macOS coordinates.
_ALIGN_ATOL = 1e-8
_ALIGN_RTOL = 2e-9
_REWARD_ATOL = 1e-6
_REWARD_RTOL = 1e-6
_INFO_ATOL = 2e-7
_INFO_RTOL = 2e-7
_LINUX_ARM64 = sys.platform == "linux" and platform.machine().lower() in (
    "aarch64",
    "arm64",
)
_LINUX_ARM64_PUSH_ALIGN_ATOL = 5e-9
_LINUX_ARM64_PUSH_REWARD_ATOL = 1.5e-6
_LINUX_ARM64_PUSH_INFO_ATOL = 1.5e-6
_LINUX_ARM64_PUSH_INFO_KEYS = {"grasp_reward", "unscaled_reward"}
_TASK_NAMES = tuple(metaworld_registration.metaworld_v3_envs)
_TASK_IDS = tuple(metaworld_registration.metaworld_v3_task_ids)
_INFO_KEYS = (
    "success",
    "near_object",
    "grasp_success",
    "grasp_reward",
    "in_place_reward",
    "obj_to_target",
    "unscaled_reward",
)


def _make_oracle(task_name: str) -> Any:
    env = ALL_V3_ENVIRONMENTS[task_name]()
    env._set_task_called = True
    env._partially_observable = True
    return env


def _sync_reset_state(oracle: Any, info: dict[str, Any]) -> np.ndarray:
    rand_vec = np.asarray(info["rand_vec0"][0], dtype=np.float64)
    random_dim = int(oracle._random_reset_space.low.size)
    oracle._freeze_rand_vec = True
    oracle._last_rand_vec = rand_vec[:random_dim].copy()
    oracle.reset()

    qpos = np.asarray(info["qpos0"][0], dtype=np.float64)[
        : oracle.data.qpos.size
    ]
    qvel = np.asarray(info["qvel0"][0], dtype=np.float64)[
        : oracle.data.qvel.size
    ]
    oracle.set_state(qpos, qvel)
    oracle.data.mocap_pos[0] = np.asarray(
        info["mocap_pos0"][0], dtype=np.float64
    )
    oracle.data.mocap_quat[0] = np.asarray(
        info["mocap_quat0"][0], dtype=np.float64
    )
    oracle.data.qacc[:] = np.asarray(info["qacc0"][0], dtype=np.float64)[
        : oracle.data.qacc.size
    ]
    oracle.data.qacc_warmstart[:] = np.asarray(
        info["qacc_warmstart0"][0], dtype=np.float64
    )[: oracle.data.qacc_warmstart.size]
    mujoco.mj_forward(oracle.model, oracle.data)
    oracle.init_tcp = np.asarray(info["init_tcp0"][0], dtype=np.float64).copy()
    oracle.init_left_pad = np.asarray(
        info["init_left_pad0"][0], dtype=np.float64
    ).copy()
    oracle.init_right_pad = np.asarray(
        info["init_right_pad0"][0], dtype=np.float64
    ).copy()
    if hasattr(oracle, "_handle_init_pos"):
        oracle._handle_init_pos = oracle._get_pos_objects().copy()

    curr_obs = oracle._get_curr_obs_combined_no_goal()
    oracle._prev_obs = curr_obs.copy()
    obs = oracle._get_obs().astype(np.float64)
    oracle._last_stable_obs = obs.copy()
    return obs


def _first_env_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs[0], dtype=np.float64)


def _align_atol(task_name: str) -> float:
    if _LINUX_ARM64 and task_name in {"push-v3", "push-wall-v3"}:
        # Linux arm64 accumulates a sub-5e-9 MuJoCo state residual in the push
        # tasks late in the rollout; other tasks keep the global 2e-9 budget.
        return _LINUX_ARM64_PUSH_ALIGN_ATOL
    return _ALIGN_ATOL


def _info_atol(task_name: str, key: str) -> float:
    if (
        _LINUX_ARM64
        and task_name in {"push-v3", "push-wall-v3"}
        and key in _LINUX_ARM64_PUSH_INFO_KEYS
    ):
        # Linux arm64 keeps observations aligned, but these reward-shaping
        # fields differ from the Python oracle by up to ~1.4e-6 after stepping.
        return _LINUX_ARM64_PUSH_INFO_ATOL
    return _INFO_ATOL


def _reward_atol(task_name: str) -> float:
    if _LINUX_ARM64 and task_name in {"push-v3", "push-wall-v3"}:
        # Linux arm64 keeps observations aligned, but the MuJoCo-backed push
        # rewards differ from the Python oracle by up to ~1.4e-6 after stepping.
        return _LINUX_ARM64_PUSH_REWARD_ATOL
    return _REWARD_ATOL


class MetaWorldAlignTest(absltest.TestCase):
    """Alignment tests for native MetaWorld against the v3.0.0 oracle."""

    def test_official_v3_registry_coverage(self) -> None:
        """EnvPool should register the official v3 registry order exactly."""
        self.assertEqual(tuple(ALL_V3_ENVIRONMENTS.keys()), _TASK_NAMES)

    def test_align_with_official_v3_rollout(self) -> None:
        """Native rollouts should match the official oracle after reset sync."""
        rng = np.random.default_rng(20260412)
        actions = rng.uniform(-1.0, 1.0, size=(_ALIGN_STEPS, 4)).astype(
            np.float32
        )
        for task_id, task_name in zip(_TASK_IDS, _TASK_NAMES, strict=True):
            with self.subTest(task_id=task_id):
                oracle = _make_oracle(task_name)
                env = make_gymnasium(task_id, num_envs=1, seed=7)
                try:
                    obs1, info1 = env.reset()
                    obs0 = _sync_reset_state(oracle, info1)
                    np.testing.assert_allclose(
                        obs0,
                        _first_env_obs(obs1),
                        atol=_align_atol(task_name),
                        rtol=_ALIGN_RTOL,
                    )

                    for step, action in enumerate(actions, start=1):
                        obs0, reward0, terminated0, truncated0, info0 = (
                            oracle.step(action)
                        )
                        obs1, reward1, terminated1, truncated1, info1 = (
                            env.step(np.asarray([action], dtype=np.float32))
                        )
                        np.testing.assert_allclose(
                            obs0,
                            _first_env_obs(obs1),
                            atol=_align_atol(task_name),
                            rtol=_ALIGN_RTOL,
                            err_msg=f"{task_id} obs step {step}",
                        )
                        np.testing.assert_allclose(
                            np.asarray(reward0, dtype=reward1.dtype),
                            reward1[0],
                            atol=_reward_atol(task_name),
                            rtol=_REWARD_RTOL,
                            err_msg=f"{task_id} reward step {step}",
                        )
                        self.assertEqual(terminated0, bool(terminated1[0]))
                        self.assertEqual(truncated0, bool(truncated1[0]))
                        for key in _INFO_KEYS:
                            np.testing.assert_allclose(
                                info0[key],
                                info1[key][0],
                                atol=_info_atol(task_name, key),
                                rtol=_INFO_RTOL,
                                err_msg=f"{task_id} info[{key}] step {step}",
                            )
                finally:
                    env.close()
                    oracle.close()


if __name__ == "__main__":
    absltest.main()
