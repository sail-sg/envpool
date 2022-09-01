"""Unit tests for football check."""

from random import random
from typing import Any
from typing import no_type_check

import numpy as np
import time
from absl.testing import absltest
from absl import logging
from envpool.football.football_envpool import _FootballEnvSpec, _FootballEnvPool

from envpool.football import (
    FootballEnvSpec,
    FootballEnvPool,
)

class _FootballTest(absltest.TestCase):
    
    def test_reset_life(self) -> None:
        np.random.seed(0)
        env = FootballEnvPool(
            FootballEnvSpec(
                FootballEnvSpec.gen_config(task="football", num_envs=1, episodic_life=True)
            )
        )
        action_num = env.action_space.n
        env.reset()
        info = env.step(np.array([0]))[-1]
        for _ in range(10000):
            _, _, done, info = env.step(np.random.randint(0, action_num, 1))
            if info["lives"][0] == 0:
                break
            else:
                self.assertFalse(info["terminated"][0])
        _, _, next_done, next_info = env.step(
            np.random.randint(0, action_num, 1)
        )
        if done[0] and next_info["lives"][0] > 0:
            self.assertTrue(info["terminated"][0])
        self.assertFalse(done[0])
        self.assertFalse(info["terminated"][0])
        while not done[0]:
            self.assertFalse(info["terminated"][0])
            _, _, done, info = env.step(np.random.randint(0, action_num, 1))
        _, _, next_done, next_info = env.step(
            np.random.randint(0, action_num, 1)
        )
        self.assertTrue(next_info["lives"][0] > 0)
        self.assertTrue(info["terminated"][0])

    def observation_space_check(self) -> None:
        obs0 = (72, 96, 16)
        env = FootballEnvPool(FootballEnvSpec(FootballEnvSpec.gen_config()))
        obs1 = env.observation_space
        np.testing.assert_allclose(obs0, obs1)

    def run_check(self) -> None:
        env = FootballEnvPool(FootballEnvSpec(FootballEnvSpec.gen_config()))
        for i in range(10):
            np.random.seed(i)
            env.action_space.seed(i)
            env.reset()
            done = False
            while not done:
                a = env.action_space.sample()
                obs, reward, done, _ = env.step(a)
                print(obs, reward, done)

if __name__ == "__main__":
    absltest.main()
