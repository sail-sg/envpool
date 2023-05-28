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

from typing import Any, Dict, List, Tuple, no_type_check

# import cv2
import gym
import numpy as np
import pygame
from absl import logging
from absl.testing import absltest
from pygame import gfxdraw

import envpool.box2d.registration  # noqa: F401
from envpool.registration import make_gym


class _Box2dEnvPoolCorrectnessTest(absltest.TestCase):

  def run_space_check(self, env0: gym.Env, env1: Any) -> None:
    """Check observation_space and action space."""
    obs0, obs1 = env0.observation_space, env1.observation_space
    np.testing.assert_allclose(obs0.shape, obs1.shape)
    act0, act1 = env0.action_space, env1.action_space
    if isinstance(act0, gym.spaces.Box):
      np.testing.assert_allclose(act0.low, act1.low)
      np.testing.assert_allclose(act0.high, act1.high)
    elif isinstance(act0, gym.spaces.Discrete):
      np.testing.assert_allclose(act0.n, act1.n)

  def test_bipedal_walker_space(self) -> None:
    env0 = gym.make("BipedalWalker-v3")
    env1 = make_gym("BipedalWalker-v3")
    self.run_space_check(env0, env1)

  def test_lunar_lander_space(self) -> None:
    env0 = gym.make("LunarLander-v2")
    env1 = make_gym("LunarLander-v2")
    self.run_space_check(env0, env1)

    env0 = gym.make("LunarLanderContinuous-v2")
    env1 = make_gym("LunarLanderContinuous-v2")
    self.run_space_check(env0, env1)

  def test_car_racing_space(self) -> None:
    env0 = gym.make("CarRacing-v2")
    env1 = make_gym("CarRacing-v2")
    self.run_space_check(env0, env1)

  @staticmethod
  def heuristic_lunar_lander_policy(
    s: np.ndarray, continuous: bool
  ) -> np.ndarray:
    angle_targ = np.clip(s[0] * 0.5 + s[2] * 1.0, -0.4, 0.4)
    hover_targ = 0.55 * np.abs(s[0])
    angle_todo = (angle_targ - s[4]) * 0.5 - s[5] * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - s[3] * 0.5

    if s[6] or s[7]:
      angle_todo = 0
      hover_todo = -(s[3]) * 0.5

    if continuous:
      a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
      a = np.clip(a, -1, 1)
    else:
      a = 0
      if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        a = 2
      elif angle_todo < -0.05:
        a = 3
      elif angle_todo > 0.05:
        a = 1
    return a

  def solve_lunar_lander(self, num_envs: int, continuous: bool) -> None:
    if continuous:
      env = make_gym("LunarLanderContinuous-v2", num_envs=num_envs)
    else:
      env = make_gym("LunarLander-v2", num_envs=num_envs)
    # each env run two episodes
    for _ in range(2):
      env_id = np.arange(num_envs)
      done = np.array([False] * num_envs)
      obs, _ = env.reset(env_id)
      rewards = np.zeros(num_envs)
      while not np.all(done):
        action = np.array(
          [self.heuristic_lunar_lander_policy(s, continuous) for s in obs]
        )
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
        self.assertTrue(abs(mean_reward - 284) < 10, (continuous, mean_reward))
      else:  # 236.898334 ± 105.832610
        self.assertTrue(abs(mean_reward - 237) < 20, (continuous, mean_reward))

  def test_lunar_lander_correctness(self, num_envs: int = 30) -> None:
    self.solve_lunar_lander(num_envs, True)
    self.solve_lunar_lander(num_envs, False)

  def solve_car_racing(
    self, num_envs: int, action: List[float], target_reward: float
  ) -> None:
    env = make_gym("CarRacing-v2", num_envs=num_envs)
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
    s: np.ndarray, h: Dict[str, float]
  ) -> Tuple[np.ndarray, Dict[str, float]]:
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
      supporting_knee_angle = min(supporting_knee_angle, SUPPORT_KNEE_ANGLE)
      knee_targ[supporting_leg] = supporting_knee_angle
      if s[supporting_s_base + 0] < 0.10:  # supporting leg is behind
        state = PUT_OTHER_DOWN
    if state == PUT_OTHER_DOWN:
      hip_targ[moving_leg] = 0.1
      knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
      knee_targ[supporting_leg] = supporting_knee_angle
      if s[moving_s_base + 4]:
        state = PUSH_OFF
        supporting_knee_angle = min(s[moving_s_base + 2], SUPPORT_KNEE_ANGLE)
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
    hs = np.array(
      [
        {
          "state": 1,
          "moving_leg": 0,
          "supporting_leg": 1,
          "supporting_knee_angle": 0.1,
        } for _ in range(num_envs)
      ]
    )
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

      ah = [self.heuristic_bipedal_walker_policy(s, h) for s, h in zip(obs, hs)]
      action = np.array([i[0] for i in ah])
      hs = np.array([i[1] for i in ah])

    mean_reward = np.mean(rewards)
    logging.info(f"{hardcore}, {np.mean(rewards):.6f} ± {np.std(rewards):.6f}")
    # the following number is from gym's 1000 episode mean reward
    if hardcore:  # -59.219390 ± 25.209768
      self.assertTrue(abs(mean_reward + 59) < 10, (hardcore, mean_reward))
    else:  # 102.647320 ± 125.075071
      self.assertTrue(abs(mean_reward - 103) < 20, (hardcore, mean_reward))

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
