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

import platform
import sys
from typing import Any

import dm_env
import mujoco
import numpy as np
from absl import logging
from absl.testing import absltest
from dm_control import suite
from packaging import version

import envpool.mujoco.dmc.registration  # noqa: F401
from envpool.registration import make_dm

_MUJOCO_V3 = version.parse(mujoco.__version__) >= version.parse("3.0.0")
_LINUX_ARM64 = sys.platform == "linux" and platform.machine().lower() in (
    "aarch64",
    "arm64",
)


class _MujocoDmcSuiteExtAlignTest(absltest.TestCase):
    def observation_atol(self, domain: str, task: str, key: str) -> float:
        if not _MUJOCO_V3:
            return 0.0
        if _LINUX_ARM64:
            return 2e-2
        if domain == "dog":
            return 2e-4
        if domain in {"quadruped", "stacker"}:
            return 1e-4
        del task, key
        return 1e-6

    def observation_rtol(self, domain: str, task: str, key: str) -> float:
        if _MUJOCO_V3 and _LINUX_ARM64:
            return 5e-3
        if domain == "dog" and key in {"foot_forces", "touch_sensors"}:
            return 1.5e-5
        del domain, task, key
        return 1e-7

    def reward_atol(self, domain: str, task: str) -> float:
        if domain in {"dog", "quadruped", "stacker"}:
            return 1e-4
        del domain, task
        return 1e-8

    def reward_rtol(self, domain: str, task: str) -> float:
        if domain in {"dog", "quadruped", "stacker"}:
            return 1e-3
        del domain, task
        return 1e-7

    def max_align_steps(self, domain: str, task: str) -> int | None:
        if domain in {"dog", "stacker"}:
            return 16
        if domain == "quadruped":
            return 64
        if domain == "lqr":
            return 1000
        del domain, task
        return None

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

    def reset_state(
        self,
        env: dm_env.Environment,
        ts: dm_env.TimeStep,
        domain: str,
        task: str,
    ) -> None:
        # manually reset, mimic initialize_episode
        with env.physics.reset_context():
            env.physics.data.qpos = ts.observation.qpos0[0]
            if hasattr(ts.observation, "qvel0"):
                env.physics.data.qvel = ts.observation.qvel0[0]
            if hasattr(ts.observation, "act0"):
                env.physics.data.act = ts.observation.act0[0]
            if domain == "lqr":
                env.physics.model.jnt_stiffness[:] = ts.observation.stiffness0[
                    0
                ]
            if domain == "quadruped" and task == "escape":
                hfield = ts.observation.hfield0[0]
                start_idx = env.physics.model.hfield_adr[0]
                env.physics.model.hfield_data[
                    start_idx : start_idx + len(hfield)
                ] = hfield
            if domain == "stacker":
                target_x, target_z = ts.observation.target0[0]
                env.physics.named.model.body_pos["target", ["x", "z"]] = (
                    target_x,
                    target_z,
                )
            if domain == "humanoid_CMU":
                env.physics.after_reset()
        if hasattr(ts.observation, "qacc_warmstart0"):
            env.physics.data.qacc_warmstart = ts.observation.qacc_warmstart0[0]

    def sample_action(
        self, action_spec: dm_env.specs.Array, domain: str
    ) -> np.ndarray:
        if domain == "lqr":
            return np.random.uniform(
                low=-1.0,
                high=1.0,
                size=action_spec.shape,
            )
        return np.random.uniform(
            low=action_spec.minimum,
            high=action_spec.maximum,
            size=action_spec.shape,
        )

    def run_align_check(
        self, env0: dm_env.Environment, env1: Any, domain: str, task: str
    ) -> None:
        logging.info(f"align check for {domain} {task}")
        max_align_steps = self.max_align_steps(domain, task)
        reward_atol = self.reward_atol(domain, task)
        reward_rtol = self.reward_rtol(domain, task)
        obs_spec, action_spec = env0.observation_spec(), env0.action_spec()
        for i in range(3):
            np.random.seed(i)
            env0.reset()
            ts = env1.reset(np.array([0]))
            self.reset_state(env0, ts, domain, task)
            logging.info(f"reset qpos {ts.observation.qpos0[0]}")
            cnt = 0
            done = False
            while not done:
                cnt += 1
                a = self.sample_action(action_spec, domain)
                # logging.info(f"{cnt} {a}")
                ts0 = env0.step(a)
                ts1 = env1.step(np.array([a]), np.array([0]))
                done = ts0.step_type == dm_env.StepType.LAST
                o0, o1 = ts0.observation, ts1.observation
                for k in obs_spec:
                    np.testing.assert_allclose(
                        o0[k],
                        getattr(o1, k)[0],
                        atol=self.observation_atol(domain, task, k),
                        rtol=self.observation_rtol(domain, task, k),
                    )
                np.testing.assert_allclose(ts0.step_type, ts1.step_type[0])
                np.testing.assert_allclose(
                    ts0.reward,
                    ts1.reward[0],
                    atol=reward_atol,
                    rtol=reward_rtol,
                )
                np.testing.assert_allclose(ts0.discount, ts1.discount[0])
                if max_align_steps is not None and cnt >= max_align_steps:
                    break

    def run_align_check_entry(self, domain: str, tasks: list[str]) -> None:
        domain_name = "".join([
            g[:1].upper() + g[1:] for g in domain.split("_")
        ])
        for task in tasks:
            task_name = "".join([
                g[:1].upper() + g[1:] for g in task.split("_")
            ])
            task_kwargs = {}
            if domain == "lqr":
                task_kwargs["time_limit"] = 30
            env0 = suite.load(domain, task, task_kwargs=task_kwargs)
            env1 = make_dm(f"{domain_name}{task_name}-v1")
            self.run_space_check(env0, env1)
            self.run_align_check(env0, env1, domain, task)

    def test_dog(self) -> None:
        self.run_align_check_entry(
            "dog", ["fetch", "run", "stand", "trot", "walk"]
        )

    def test_humanoid_CMU(self) -> None:
        if sys.platform == "darwin" and _MUJOCO_V3:
            self.skipTest(
                "MuJoCo humanoid alignment is numerically unstable on macOS"
            )
        self.run_align_check_entry("humanoid_CMU", ["stand", "walk", "run"])

    def test_lqr(self) -> None:
        self.run_align_check_entry("lqr", ["lqr_2_1", "lqr_6_2"])

    def test_quadruped(self) -> None:
        if sys.platform == "darwin":
            self.skipTest(
                "dm_control quadruped reset requires a working GLFW context on macOS"
            )
        self.run_align_check_entry(
            "quadruped", ["escape", "fetch", "run", "walk"]
        )

    def test_stacker(self) -> None:
        self.run_align_check_entry("stacker", ["stack_2", "stack_4"])


if __name__ == "__main__":
    absltest.main()
