# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inspect independently sampled components without synchronizing either pool."""

from contextlib import closing
from typing import Any

import numpy as np
from absl.testing import absltest, parameterized

import envpool.gymnax.registration  # noqa: F401
from envpool.gymnax.oracle import TASK_IDS, make_oracle, speeds
from envpool.python.seed_test_utils import check_seeded_rollouts
from envpool.registration import make_gymnasium

_FIXED = {"DeepSea", "DiscountingChain", "Pong", "SpaceInvaders"}
_CASES = [(task.split("-")[0], task) for task in TASK_IDS]


def _samples(name: str, pool: Any, ref: Any) -> np.ndarray:
    """One independent sample per slot. Scale bounded components to [0, 1]."""
    obs, info = pool.reset()
    n = len(obs)
    h = info["hidden"]

    def step(a: Any) -> Any:
        return pool.step(np.broadcast_to(a, (n,)).astype(np.int32))

    if name == "PointRobot":
        goal = h[:, :2]
        np.testing.assert_allclose(np.linalg.norm(goal, axis=1), 1, atol=1e-7)
        return np.c_[
            np.arctan2(goal[:, 1], goal[:, 0]) / np.pi,
            np.arctan2(obs[:, 1], obs[:, 0]) / np.pi,
            np.linalg.norm(obs[:, :2], axis=1),
        ]
    if name == "Reacher":
        return (
            np.c_[
                np.arctan2(obs[:, 2:4], obs[:, :2]) / (2 * np.pi),
                obs[:, 4:6] / 2,
                obs[:, 6:] / 4,
            ]
            + 0.5
        )
    if name == "Swimmer":
        assert (obs[:, 12:14] == 0).all()
        return np.c_[obs[:, :12], obs[:, 14:]] / 2 + 0.5
    if name == "MetaMaze":
        goal = h[:, 0].astype(int)
        assert np.asarray(ref.env_map).flat[goal].all()
        return (np.column_stack(divmod(goal, 11)) - 1) / 8
    if name == "Snake":
        assert not (obs[..., 1] * obs[..., 2]).any()
        heads, apples = (np.argwhere(obs[..., k])[:, 1:] for k in (1, 2))
        return np.c_[heads / 9, apples / 9, h[:, 0] / 3]
    if name == "MemoryChain":
        for t in range(6):
            reward = step(t % 2)[1]
        np.testing.assert_array_equal(reward, obs[:, 2])
        return (obs[:, 2:] + 1) / 2
    if name == "UmbrellaChain":
        rewards = [step(t % 2)[1] for t in range(10)]
        np.testing.assert_array_equal(rewards[-1], 1 - 2 * obs[:, 0])
        return np.c_[obs[:, :2], (np.stack(rewards[:-1], axis=1) + 1) / 2]
    if name == "BernoulliBandit":
        good = (h[:, 0] < 0.5).astype(int)
        np.testing.assert_allclose(h[:, 0], np.where(good, 0.1, 0.9))
        return np.c_[good, step(good)[1], step(1 - good)[1]]
    if name == "GaussianBandit":
        mu = h[:, 0]
        return np.c_[mu + 1, (step(1)[1] - mu) / 0.1, (step(1)[1] - mu) / 0.1]
    if name == "SimpleBandit":
        arms = np.broadcast_to(np.linspace(0, 1, ref.num_actions), h.shape)
        np.testing.assert_allclose(np.sort(h, axis=1), arms)
        a = np.arange(n) % ref.num_actions
        np.testing.assert_array_equal(step(a)[1], h[np.arange(n), a])
        return h
    if name == "MNISTBandit":
        idx = h[:, 0].astype(int)
        images = np.asarray(ref.images)[idx].astype(np.float32) / 255
        np.testing.assert_allclose(obs, images, atol=1e-7)
        wrong = np.arange(n) % 2
        label = np.asarray(ref.labels)[idx]
        np.testing.assert_array_equal(
            step((label + wrong) % 10)[1], 1 - 2 * wrong
        )
        return idx[:, None] / (ref.num_data - 1)
    if name == "Breakout":
        x = obs[:, 3, :, 1].argmax(axis=1, keepdims=True)
        assert np.isin(x, [0, 9]).all()
        return x / 9
    if name == "Freeway":
        v = speeds(obs)
        return np.c_[(np.abs(v) - 1) / 4, v > 0]

    # These tasks reset to fixed observations. Observe actual movement and
    # spawns under identical actions across all seeds, resets and slots.
    if name == "FourRooms":
        pos = step(1)[0][:, :2]
        return (pos[:, None, :] == [[3, 1], [4, 2], [5, 1], [4, 1]]).all(
            axis=-1
        )
    if name == "Asterix":
        for _ in range(11):
            obs = step(0)[0]
        i, y, x = np.nonzero(obs[..., 1] + obs[..., 3])
        np.testing.assert_array_equal(i, np.arange(n))
        return np.c_[(y - 1) / 7, x < 5, obs[i, y, x, 3]]
    if name == "Seaquest":
        for _ in range(21):
            obs = step(0)[0]
        i, y, x = np.nonzero(obs[..., 5] + obs[..., 6])
        np.testing.assert_array_equal(i, np.arange(n))
        enemy = np.c_[(y - 1) / 7, x < 5, obs[i, y, x, 6]]
        for _ in range(10):
            obs = step(0)[0]
        i, y, x = np.nonzero(obs[..., 9])
        np.testing.assert_array_equal(i, np.arange(n))
        return np.c_[enemy, (y - 1) / 7, x < 5]
    raise AssertionError(name)


class _SamplingTest(parameterized.TestCase):
    @parameterized.named_parameters(_CASES)
    def test_repeatable_rollouts(self, task: str) -> None:
        name = task.split("-")[0]
        info = ("hidden",) if name in {"MetaMaze", "PointRobot"} else ()
        check_seeded_rollouts(
            self,
            "Gymnax/" + task,
            info_keys=info,
            expect_different=name not in _FIXED,
        )

    @parameterized.named_parameters([c for c in _CASES if c[0] not in _FIXED])
    def test_samples(self, task: str) -> None:
        name = task.split("-")[0]
        ref, _ = make_oracle(task)

        def draws(seed: int) -> np.ndarray:
            with closing(
                make_gymnasium(
                    "Gymnax/" + task, num_envs=64, num_threads=2, seed=seed
                )
            ) as pool:
                return np.stack([_samples(name, pool, ref) for _ in range(8)])

        a, b = draws(17), draws(971)
        np.testing.assert_array_equal(a, draws(17))
        self.assertTrue((a != b).any(axis=(0, 1)).all(), "different seeds")
        self.assertTrue(
            (a[1:] != a[:1]).any(axis=(0, 1)).all(), "successive resets"
        )
        self.assertTrue(
            (a[:, 1:] != a[:, :1]).any(axis=(0, 1)).all(), "parallel slots"
        )
        values = np.concatenate([a, b]).reshape(-1, a.shape[-1])
        # 1024 samples per component. The bounds on the mean allow over
        # five standard errors even for coin flips. Also reject collapsed
        # ranges and coupled random variables (e.g., a goal tied to a pose).
        if name == "GaussianBandit":
            np.testing.assert_allclose(values.mean(axis=0), 0, atol=0.17)
            np.testing.assert_allclose(values.std(axis=0), 1, atol=0.14)
        else:
            means = {
                "BernoulliBandit": [0.5, 0.9, 0.1],
                "Asterix": [0.5, 0.5, 1 / 3],
                "Seaquest": [0.5, 0.5, 1 / 3, 0.5, 0.5],
                "FourRooms": [1 / 9, 2 / 3, 1 / 9, 1 / 9],
            }
            self.assertGreaterEqual(values.min(), -1e-6)
            self.assertLessEqual(values.max(), 1 + 1e-6)
            np.testing.assert_allclose(
                values.mean(axis=0), means.get(name, 0.5), atol=0.08
            )
            self.assertGreater(values.std(axis=0).min(), 0.22)
        if name == "FourRooms":
            np.testing.assert_array_equal(values.sum(axis=1), 1)
        elif values.shape[1] > 1:
            corr = np.corrcoef(values.T)
            np.fill_diagonal(corr, 0)
            self.assertLess(np.abs(corr).max(), 0.25)


if __name__ == "__main__":
    absltest.main()
