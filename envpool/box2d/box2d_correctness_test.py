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
"""Unit tests for box2d environments correctness check."""

import importlib.machinery
import importlib.util
import platform
import re
import sys
import types
from pathlib import Path
from typing import Any, no_type_check

import gymnasium as gym
import numpy as np
import pygame
from absl import logging
from absl.testing import absltest
from pygame import gfxdraw

import envpool.box2d.registration  # noqa: F401
from envpool.registration import make_gym

_LINUX_ARM64 = sys.platform == "linux" and platform.machine().lower() in (
    "aarch64",
    "arm64",
)
_BOX2D_SWIGCONSTANT_RE = re.compile(r"_Box2D\.(\w+_swigconstant)\(")


def _patch_box2d_swigconstant_shims(module: Any, pathname: str) -> None:
    wrapper_path = Path(pathname).with_name("Box2D.py")
    try:
        names = set(_BOX2D_SWIGCONSTANT_RE.findall(wrapper_path.read_text()))
    except OSError:
        return
    for attr in names:
        if not hasattr(module, attr):
            # Box2D 2.3.2 ships a stale Python wrapper on Linux arm64, while
            # the source build regenerates only the C extension with modern
            # SWIG. The newer extension already exposes the constants
            # directly, so the legacy *_swigconstant hooks can be harmless
            # no-ops.
            setattr(module, attr, lambda _target, _attr=attr: None)


def _install_imp_compat() -> None:
    try:
        import imp  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    compat_imp: Any = types.ModuleType("imp")
    compat_imp.C_EXTENSION = 3

    def find_module(
        name: str, path: Any = None
    ) -> tuple[Any, str, tuple[str, str, int]]:
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.origin is None:
            raise ImportError(name)
        return (
            open(spec.origin, "rb"),
            spec.origin,
            ("", "rb", compat_imp.C_EXTENSION),
        )

    def load_module(
        name: str, file: Any, pathname: str, description: Any
    ) -> Any:
        del file, description
        module = sys.modules.get(name)
        if module is not None:
            return module
        spec = importlib.util.spec_from_file_location(name, pathname)
        if spec is None or spec.loader is None:
            raise ImportError(pathname)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if name == "_Box2D":
            _patch_box2d_swigconstant_shims(module, pathname)
        return module

    compat_imp.find_module = find_module
    compat_imp.load_module = load_module
    sys.modules["imp"] = compat_imp


_install_imp_compat()


class _Box2dEnvPoolCorrectnessTest(absltest.TestCase):
    def run_space_check(self, env0: gym.Env, env1: Any) -> None:
        """Check observation_space and action space."""
        obs0, obs1 = env0.observation_space, env1.observation_space
        self.assertEqual(obs0.shape, obs1.shape)
        act0, act1 = env0.action_space, env1.action_space
        if isinstance(act0, gym.spaces.Box):
            np.testing.assert_allclose(act0.low, act1.low)
            np.testing.assert_allclose(act0.high, act1.high)
        elif isinstance(act0, gym.spaces.Discrete):
            self.assertEqual(act0.n, act1.n)

    def test_bipedal_walker_space(self) -> None:
        env0 = gym.make("BipedalWalker-v3")
        env1 = make_gym("BipedalWalker-v3")
        self.run_space_check(env0, env1)

    def test_lunar_lander_space(self) -> None:
        env0 = gym.make("LunarLander-v3")
        env1 = make_gym("LunarLander-v3")
        self.run_space_check(env0, env1)

        env0 = gym.make("LunarLanderContinuous-v3")
        env1 = make_gym("LunarLanderContinuous-v3")
        self.run_space_check(env0, env1)

    def test_car_racing_space(self) -> None:
        env0 = gym.make("CarRacing-v3")
        env1 = make_gym("CarRacing-v3")
        self.run_space_check(env0, env1)

    @staticmethod
    def heuristic_lunar_lander_policy(s: np.ndarray, continuous: bool) -> Any:
        angle_targ = np.clip(s[0] * 0.5 + s[2] * 1.0, -0.4, 0.4)
        hover_targ = 0.55 * np.abs(s[0])
        angle_todo = (angle_targ - s[4]) * 0.5 - s[5] * 1.0
        hover_todo = (hover_targ - s[1]) * 0.5 - s[3] * 0.5

        if s[6] or s[7]:
            angle_todo = 0
            hover_todo = -(s[3]) * 0.5

        if continuous:
            action = np.array([hover_todo * 20 - 1, -angle_todo * 20])
            return np.clip(action, -1, 1)
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            return 2
        if angle_todo < -0.05:
            return 3
        if angle_todo > 0.05:
            return 1
        return 0

    def solve_lunar_lander(self, num_envs: int, continuous: bool) -> None:
        if continuous:
            env = make_gym("LunarLanderContinuous-v3", num_envs=num_envs)
        else:
            env = make_gym("LunarLander-v3", num_envs=num_envs)
        # each env run two episodes
        for _ in range(2):
            env_id = np.arange(num_envs)
            done = np.array([False] * num_envs)
            obs, _ = env.reset(env_id)
            rewards = np.zeros(num_envs)
            while not np.all(done):
                action = np.array([
                    self.heuristic_lunar_lander_policy(s, continuous)
                    for s in obs
                ])
                obs, rew, terminated, truncated, info = env.step(action, env_id)
                done = np.logical_or(terminated, truncated)
                env_id = info["env_id"]
                rewards[env_id] += rew
                obs = obs[~done]
                env_id = env_id[~done]
            mean_reward = np.mean(rewards)
            logging.info(
                f"{continuous}, {np.mean(rewards):.6f} ± {np.std(rewards):.6f}"
            )
            # the following number is from gym's 1000 episode mean reward
            if continuous:  # 283.872619 ± 18.881830
                if sys.platform == "darwin" or _LINUX_ARM64:
                    # Gymnasium's current macOS Box2D stack lands a bit lower
                    # than the historical Linux-derived baseline, and Linux
                    # arm64 exhibits the same drift under the heuristic policy.
                    self.assertTrue(
                        abs(mean_reward - 282) < 15, (continuous, mean_reward)
                    )
                else:
                    self.assertTrue(
                        abs(mean_reward - 284) < 10, (continuous, mean_reward)
                    )
            else:  # 236.898334 ± 105.832610
                if sys.platform == "darwin" or _LINUX_ARM64:
                    self.assertTrue(
                        abs(mean_reward - 221) < 25, (continuous, mean_reward)
                    )
                else:
                    self.assertTrue(
                        abs(mean_reward - 237) < 20, (continuous, mean_reward)
                    )

    def test_lunar_lander_correctness(self, num_envs: int = 30) -> None:
        self.solve_lunar_lander(num_envs, True)
        self.solve_lunar_lander(num_envs, False)

    def solve_car_racing(
        self, num_envs: int, action: list[float], target_reward: float
    ) -> None:
        env = make_gym("CarRacing-v3", num_envs=num_envs)
        max_episode_steps = 100

        env_id = np.arange(num_envs)
        done = np.array([False] * num_envs)
        obs = env.reset(env_id)
        rewards = np.zeros(num_envs)
        action = np.tile(action, (num_envs, 1))
        for _ in range(max_episode_steps):
            obs, rew, terminated, truncated, info = env.step(action, env_id)
            env_id = info["env_id"]
            rewards[env_id] += rew
            # cv2.imwrite("/tmp/car_racing-{}.jpg".format(i), obs[0])
            if np.all(done):
                break
            obs = obs[~done]
            env_id = env_id[~done]
        mean_reward = np.mean(rewards)
        logging.info(f"{np.mean(rewards):.6f} ± {np.std(rewards):.6f}")

        self.assertTrue(abs(target_reward - mean_reward) < 1, (mean_reward))

    def test_car_racing_correctness(
        self, num_envs: int = 100, render: bool = False
    ) -> None:
        if render:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((600, 400))
            self.clock = pygame.time.Clock()
        self.solve_car_racing(num_envs, [0, 0.5, 0], 65)
        self.solve_car_racing(num_envs, [0.1, 0.3, 0], 18.5)
        self.solve_car_racing(num_envs, [0, 0.7, 0.1], 42.7)

    @staticmethod
    @no_type_check
    def heuristic_bipedal_walker_policy(
        s: np.ndarray, h: dict[str, float]
    ) -> tuple[np.ndarray, dict[str, float]]:
        STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1, 2, 3
        SPEED = 0.29  # will fall forward on higher speed
        SUPPORT_KNEE_ANGLE = 0.1
        state = h["state"]
        moving_leg = h["moving_leg"]
        supporting_leg = h["supporting_leg"]
        supporting_knee_angle = h["supporting_knee_angle"]

        moving_s_base = 4 + 5 * moving_leg
        supporting_s_base = 4 + 5 * supporting_leg

        hip_targ = [0.0, 0.0]  # -0.8 .. +1.1
        knee_targ = [0.0, 0.0]  # -0.6 .. +0.9
        hip_todo = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if state == STAY_ON_ONE_LEG:
            hip_targ[moving_leg] = 1.1
            knee_targ[moving_leg] = -0.6
            supporting_knee_angle += 0.03
            if s[2] > SPEED:
                supporting_knee_angle += 0.03
            supporting_knee_angle = min(
                supporting_knee_angle, SUPPORT_KNEE_ANGLE
            )
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[supporting_s_base + 0] < 0.10:  # supporting leg is behind
                state = PUT_OTHER_DOWN
        if state == PUT_OTHER_DOWN:
            hip_targ[moving_leg] = 0.1
            knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[moving_s_base + 4]:
                state = PUSH_OFF
                supporting_knee_angle = min(
                    s[moving_s_base + 2], SUPPORT_KNEE_ANGLE
                )
        if state == PUSH_OFF:
            knee_targ[moving_leg] = supporting_knee_angle
            knee_targ[supporting_leg] = 1.0
            if s[supporting_s_base + 2] > 0.88 or s[2] > 1.2 * SPEED:
                state = STAY_ON_ONE_LEG
                moving_leg = 1 - moving_leg
                supporting_leg = 1 - moving_leg

        if hip_targ[0]:
            hip_todo[0] = 0.9 * (hip_targ[0] - s[4]) - 0.25 * s[5]
        if hip_targ[1]:
            hip_todo[1] = 0.9 * (hip_targ[1] - s[9]) - 0.25 * s[10]
        if knee_targ[0]:
            knee_todo[0] = 4.0 * (knee_targ[0] - s[6]) - 0.25 * s[7]
        if knee_targ[1]:
            knee_todo[1] = 4.0 * (knee_targ[1] - s[11]) - 0.25 * s[12]

        hip_todo[0] -= 0.9 * (0 - s[0]) - 1.5 * s[1]  # PID to keep head strait
        hip_todo[1] -= 0.9 * (0 - s[0]) - 1.5 * s[1]
        knee_todo[0] -= 15.0 * s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0 * s[3]

        a = np.array([hip_todo[0], knee_todo[0], hip_todo[1], knee_todo[1]])
        a = np.clip(0.5 * a, -1.0, 1.0)
        return a, {
            "state": state,
            "moving_leg": moving_leg,
            "supporting_leg": supporting_leg,
            "supporting_knee_angle": supporting_knee_angle,
        }

    def solve_bipedal_walker(
        self, num_envs: int, hardcore: bool, render: bool
    ) -> None:
        if hardcore:
            env = make_gym("BipedalWalkerHardcore-v3", num_envs=num_envs)
        else:
            env = make_gym("BipedalWalker-v3", num_envs=num_envs)
        max_episode_steps = env.spec.config.max_episode_steps
        hs = np.array([
            {
                "state": 1,
                "moving_leg": 0,
                "supporting_leg": 1,
                "supporting_knee_angle": 0.1,
            }
            for _ in range(num_envs)
        ])
        env_id = np.arange(num_envs)
        done = np.array([False] * num_envs)
        obs, _ = env.reset(env_id)
        rewards = np.zeros(num_envs)
        action = np.zeros([num_envs, 4])
        for _ in range(max_episode_steps):
            obs, rew, terminated, truncated, info = env.step(action, env_id)
            done = np.logical_or(terminated, truncated)
            if render:
                self.render_bpw(info)
            env_id = info["env_id"]
            rewards[env_id] += rew
            if np.all(done):
                break
            obs = obs[~done]
            env_id = env_id[~done]
            hs = hs[~done]

            ah = [
                self.heuristic_bipedal_walker_policy(s, h)
                for s, h in zip(obs, hs, strict=False)
            ]
            action = np.array([i[0] for i in ah])
            hs = np.array([i[1] for i in ah])

        mean_reward = np.mean(rewards)
        logging.info(
            f"{hardcore}, {np.mean(rewards):.6f} ± {np.std(rewards):.6f}"
        )
        # These heuristic rollout baselines track the current Gymnasium/Box2D
        # stack. BipedalWalker is sensitive to tiny long-horizon numerical
        # drift under this controller, so keep this as a loose distribution
        # sanity check; step-level alignment is covered in box2d_align_test.
        if hardcore:  # -59.219390 ± 25.209768
            self.assertTrue(abs(mean_reward + 59) < 10, (hardcore, mean_reward))
        else:  # 110.241196 ± 131.039104 on gymnasium 1.2.3 / box2d 2.3.10
            self.assertTrue(
                abs(mean_reward - 110) < 30, (hardcore, mean_reward)
            )

    def render_bpw(self, info: dict) -> None:
        SCALE = 30.0
        VIEWPORT_W = 600
        VIEWPORT_H = 400
        scroll = info["scroll"][0]
        surf = pygame.Surface((VIEWPORT_W + scroll * SCALE, VIEWPORT_H))
        pygame.transform.scale(surf, (SCALE, SCALE))
        pygame.draw.polygon(
            surf,
            color=(215, 215, 255),
            points=[
                (scroll * SCALE, 0),
                (scroll * SCALE + VIEWPORT_W, 0),
                (scroll * SCALE + VIEWPORT_W, VIEWPORT_H),
                (scroll * SCALE, VIEWPORT_H),
            ],
        )
        for p in info["path2"][0]:
            c = (0, 255, 0)
            pygame.draw.aaline(surf, start_pos=p[0], end_pos=p[1], color=c)
        for p in info["path4"][0]:
            c = (255, 255, 255)
            p = p.tolist()
            pygame.draw.polygon(surf, color=c, points=p)
            gfxdraw.aapolygon(surf, p, c)
            c = (153, 153, 153)
            p.append(p[0])
            pygame.draw.polygon(surf, color=c, points=p, width=1)
            gfxdraw.aapolygon(surf, p, c)
        for p in info["path5"][0]:
            c = (127, 51, 229)
            p = p.tolist()
            pygame.draw.polygon(surf, color=c, points=p)
            gfxdraw.aapolygon(surf, p, c)
            c = (76, 76, 127)
            p.append(p[0])
            pygame.draw.polygon(surf, color=c, points=p, width=1)
            gfxdraw.aapolygon(surf, p, c)
        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (-scroll * SCALE, 0))
        pygame.event.pump()
        self.clock.tick(50)
        pygame.display.flip()

    def test_bipedal_walker_correctness(
        self, num_envs: int = 100, render: bool = False
    ) -> None:
        if render:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((600, 400))
            self.clock = pygame.time.Clock()
        self.solve_bipedal_walker(num_envs, True, render)
        self.solve_bipedal_walker(num_envs, False, render)


if __name__ == "__main__":
    absltest.main()
