# Copyright 2023 Garena Online Private Limited
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
"""Unit tests for Procgen environments."""

# import cv2
import numpy as np
from absl import logging
from absl.testing import absltest

from envpool.procgen.registration import distribution, procgen_game_config
from envpool.registration import make_gym

try:
    from procgen import ProcgenGym3Env
except ImportError:
    ProcgenGym3Env = None


class _ProcgenEnvPoolTest(absltest.TestCase):
    def procgen_oracle_check(
        self,
        task_id: str,
        env_name: str,
        dist_value: int,
        seed: int = 0,
        total: int = 200,
    ) -> None:
        if ProcgenGym3Env is None:
            self.skipTest("upstream procgen is not installed")

        logging.info(f"procgen oracle check for {task_id}")
        envpool_env = make_gym(
            task_id,
            num_envs=1,
            batch_size=1,
            num_threads=1,
            seed=seed,
            channel_first=False,
        )
        procgen_env = None
        rng = np.random.default_rng(seed)
        try:
            procgen_env = ProcgenGym3Env(
                num=1,
                env_name=env_name,
                distribution_mode=distribution[dist_value].lower(),
                rand_seed=seed,
                num_threads=1,
            )
            envpool_obs, envpool_info = envpool_env.reset()
            procgen_rew, procgen_obs, procgen_first = procgen_env.observe()
            procgen_info = procgen_env.get_info()[0]
            self.assertTrue(procgen_first[0])
            self.assertEqual(procgen_rew[0], 0.0)
            np.testing.assert_array_equal(envpool_obs, procgen_obs["rgb"])
            self.assertEqual(
                envpool_info["level_seed"][0], procgen_info["level_seed"]
            )
            self.assertEqual(
                envpool_info["prev_level_seed"][0],
                procgen_info["prev_level_seed"],
            )
            self.assertEqual(
                envpool_info["prev_level_complete"][0],
                procgen_info["prev_level_complete"],
            )

            for _ in range(total):
                action = rng.integers(15, size=(1,), dtype=np.int32)
                (
                    envpool_obs,
                    envpool_rew,
                    envpool_term,
                    envpool_trunc,
                    envpool_info,
                ) = envpool_env.step(action)
                procgen_env.act(action)
                procgen_rew, procgen_obs, procgen_first = procgen_env.observe()
                procgen_info = procgen_env.get_info()[0]

                np.testing.assert_array_equal(envpool_rew, procgen_rew)
                self.assertEqual(
                    bool(envpool_term[0] or envpool_trunc[0]),
                    bool(procgen_first[0]),
                )
                # Upstream procgen auto-resets inside terminal transitions while
                # EnvPool resets on the next step, so terminal frames are offset.
                if procgen_first[0]:
                    break

                np.testing.assert_array_equal(envpool_obs, procgen_obs["rgb"])
                self.assertEqual(
                    envpool_info["level_seed"][0], procgen_info["level_seed"]
                )
                self.assertEqual(
                    envpool_info["prev_level_seed"][0],
                    procgen_info["prev_level_seed"],
                )
                self.assertEqual(
                    envpool_info["prev_level_complete"][0],
                    procgen_info["prev_level_complete"],
                )
        finally:
            envpool_env.close()
            if procgen_env is not None:
                procgen_env.close()

    def deterministic_check(
        self,
        task_id: str,
        num_envs: int = 4,
        total: int = 200,
        expect_different_seed_obs: bool = True,
    ) -> None:
        logging.info(f"deterministic check for {task_id}")
        env0 = make_gym(task_id, num_envs=num_envs, seed=0)
        env1 = make_gym(task_id, num_envs=num_envs, seed=0)
        env2 = make_gym(task_id, num_envs=num_envs, seed=1)
        act_space = env0.action_space
        for _ in range(total):
            action = np.array([act_space.sample() for _ in range(num_envs)])
            obs0 = env0.step(action)[0]
            obs1 = env1.step(action)[0]
            obs2 = env2.step(action)[0]
            np.testing.assert_allclose(obs0, obs1)
            if expect_different_seed_obs:
                self.assertFalse(np.allclose(obs0, obs2))
            else:
                np.testing.assert_allclose(obs0, obs2)

    def test_deterministic(self) -> None:
        for env_name, _, dist_mode in procgen_game_config:
            for dist_value in dist_mode:
                task_id = (
                    f"{env_name.capitalize()}{distribution[dist_value]}-v0"
                )
                self.deterministic_check(
                    task_id,
                    expect_different_seed_obs=(dist_value != 20),
                )

    def test_procgen_oracle(self) -> None:
        for env_name, _, dist_mode in procgen_game_config:
            for dist_value in dist_mode:
                task_id = (
                    f"{env_name.capitalize()}{distribution[dist_value]}-v0"
                )
                self.procgen_oracle_check(task_id, env_name, dist_value)

    def test_align(self) -> None:
        task_id = "CoinrunHard-v0"
        seed = 0
        env = make_gym(task_id, seed=seed, channel_first=False)
        env.action_space.seed(seed)
        done = [False]
        cnt = sum_reward = sum_obs = 0
        while not done[0]:
            cnt += 1
            act = env.action_space.sample()
            obs, rew, term, trunc, info = env.step(np.array([act]))
            sum_obs = obs[0].astype(int) + sum_obs
            done = term | trunc
            sum_reward += rew[0]
            # cv2.imwrite(f"/tmp/envpool/{cnt}.png", obs[0])
            # print(f"{cnt=} {obs.sum()=} {done=} {rew=} {info=}")
        self.assertEqual(sum_reward, 0)
        self.assertEqual(rew[0], 0)
        self.assertEqual(cnt, 1001)
        self.assertEqual(info["level_seed"][0], 1263407424)
        self.assertEqual(info["prev_level_complete"][0], 0)
        pixel_mean_ref = [144.37373564, 150.09581459, 177.80634478]
        pixel_mean = (sum_obs / cnt).mean(axis=0).mean(axis=0)  # type: ignore
        # Apple Silicon renders the same trajectory with sub-1e-3 pixel drift.
        np.testing.assert_allclose(
            pixel_mean,
            pixel_mean_ref,
            rtol=1e-5,
            atol=1e-3,
        )

    def test_channel_first(
        self,
        task_id: str = "CoinrunHard-v0",
        seed: int = 0,
        total: int = 1000,
    ) -> None:
        env1 = make_gym(task_id, seed=seed, channel_first=True)
        env2 = make_gym(task_id, seed=seed, channel_first=False)
        self.assertEqual(env1.observation_space.shape, (3, 64, 64))
        self.assertEqual(env2.observation_space.shape, (64, 64, 3))
        for _ in range(total):
            act = env1.action_space.sample()
            obs1 = env1.step(np.array([act]))[0][0]
            obs2 = env2.step(np.array([act]))[0][0]
            self.assertEqual(obs1.shape, (3, 64, 64))
            self.assertEqual(obs2.shape, (64, 64, 3))
            np.testing.assert_allclose(obs1, obs2.transpose(2, 0, 1))


if __name__ == "__main__":
    absltest.main()
