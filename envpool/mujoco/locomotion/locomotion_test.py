# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exercise every native task through EnvPool's batched public APIs."""

from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

import numpy as np
from absl.testing import absltest, parameterized
from envpool.mujoco.locomotion.locomotion_envpool import TASKS

import envpool.mujoco.locomotion.registration  # noqa: F401
from envpool.mujoco.render_test_utils import assert_rgb_images
from envpool.registration import make_dm, make_gymnasium


def assert_observations(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    task: str,
    context: str,
) -> None:
    """Compare native observations with the existing camera budgets."""
    np.testing.assert_equal(sorted(actual), sorted(expected))
    for key, value in actual.items():
        if key == "walker/egocentric_camera":
            assert_rgb_images(value, expected[key], f"{task}, {context}")
        else:
            np.testing.assert_array_equal(
                value, expected[key], err_msg=f"{context}, {key}"
            )


def check_reset_randomization(
    test: absltest.TestCase,
    task: str,
    components: Callable[[Any, int, dict], dict] | None = None,
) -> None:
    """Reject frozen resets (#432), including independently random fields."""
    sequences = []
    for seed in (11, 11, 43):
        with ExitStack() as stack:
            pool = make_gymnasium(
                f"dm_control/locomotion/{task}",
                num_envs=4,
                num_threads=2,
                seed=seed,
            )
            stack.callback(pool.close)
            resets = []
            for _ in range(8):
                obs, info = pool.reset()
                # Compare complete matches, not players on opposite teams.
                rows = []
                for env_id in range(4):
                    mask = info["players"]["env_id"] == env_id
                    row = {key: value[mask] for key, value in obs.items()}
                    rows.append(
                        row
                        if components is None
                        else components(pool, env_id, row)
                    )
                resets.append(rows)
            sequences.append(resets)

    def differs(left: dict, right: dict, field: str | None = None) -> bool:
        if field is not None:
            left, right = {field: left[field]}, {field: right[field]}
        try:
            assert_observations(left, right, task, "reset variation")
        except AssertionError:
            return True
        return False

    for reset in range(8):
        for env_id in range(4):
            assert_observations(
                sequences[0][reset][env_id],
                sequences[1][reset][env_id],
                task,
                f"{task}, seed replay, reset {reset}, env {env_id}",
            )
    # No IDs, counters, RNG state, or reset-time oracle synchronization can
    # establish randomization. Ignore the same CGL noise used in replay checks.
    fields = [None, *sequences[0][0][0]] if components else [None]
    for field in fields:
        with test.subTest(task=task, component=field):
            test.assertTrue(
                any(
                    differs(sequences[0][r][e], sequences[2][r][e], field)
                    for r in range(8)
                    for e in range(4)
                ),
                "different seeds must change state",
            )
            test.assertTrue(
                any(
                    differs(resets[r][0], resets[r][e], field)
                    for resets in sequences
                    for r in range(8)
                    for e in range(1, 4)
                ),
                "parallel environments must differ",
            )
            test.assertTrue(
                any(
                    differs(resets[0][e], resets[r][e], field)
                    for resets in sequences
                    for r in range(1, 8)
                    for e in range(4)
                ),
                "successive resets must change state",
            )
    # Every stream must advance, but discrete components may legitimately
    # repeat individual samples (e.g. either assignment of target colors).
    for pool_index, resets in enumerate(sequences):
        for env_id in range(4):
            test.assertTrue(
                any(
                    differs(resets[0][env_id], resets[r][env_id])
                    for r in range(1, 8)
                ),
                f"{task}, frozen reset stream: pool {pool_index}, env {env_id}",
            )


class LocomotionTest(parameterized.TestCase):
    """Check public batching, rendering, and repeatable complete rollouts."""

    @parameterized.named_parameters((task, task) for task in TASKS)
    def test_seeded_resets(self, task: str) -> None:
        """Keep default seed/reset/parallel variation in installed-wheel tests."""
        check_reset_randomization(self, task)

    @parameterized.named_parameters((task, task) for task in TASKS)
    def test_deterministic_rollout(self, task: str) -> None:
        """Replay actions across worker counts and multiple episode resets."""
        kwargs = dict(
            num_envs=2,
            seed=[5, 9],
            max_episode_steps=96,
            render_mode="rgb_array",
            render_width=96,
            render_height=80,
        )
        left = make_gymnasium(
            f"dm_control/locomotion/{task}", num_threads=1, **kwargs
        )
        right = make_gymnasium(
            f"dm_control/locomotion/{task}", num_threads=2, **kwargs
        )
        random = np.random.RandomState(123)
        left_obs, left_info = left.reset()
        right_obs, right_info = right.reset()
        saw_end = False
        for step in range(195):
            li = np.argsort(left_info["players"]["env_id"], kind="stable")
            ri = np.argsort(right_info["players"]["env_id"], kind="stable")
            assert_observations(
                {key: value[li] for key, value in left_obs.items()},
                {key: value[ri] for key, value in right_obs.items()},
                task,
                f"{task}: step {step}",
            )
            for key in left_obs:
                self.assertTrue(
                    left.observation_space[key].contains(left_obs[key][0]), key
                )
            if step in (0, 32, 97, 194):
                a = left.render(env_ids=[1, 0])
                b = right.render(env_ids=[1, 0])
                assert a is not None and b is not None
                self.assertEqual(a.shape, (2, 80, 96, 3))
                self.assertEqual(a.dtype, np.uint8)
                assert_rgb_images(a, b, f"{task}, render step {step}")
                repeat = left.render(env_ids=[1])
                assert repeat is not None
                assert_rgb_images(
                    a[:1], repeat, f"{task}, repeated render step {step}"
                )
            action = random.uniform(
                -0.25, 0.25, (len(li), *left.action_space.shape)
            )
            if step % 3 == 0:
                action *= np.sin(step * 0.17)
            left_obs, lr, lt, lx, left_info = left.step(
                action[np.argsort(li)], env_id=left_info["env_id"]
            )
            right_obs, rr, rt, rx, right_info = right.step(
                action[np.argsort(ri)], env_id=right_info["env_id"]
            )
            le = np.argsort(left_info["env_id"])
            re = np.argsort(right_info["env_id"])
            np.testing.assert_array_equal(lt[le], rt[re])
            np.testing.assert_array_equal(lx[le], rx[re])
            np.testing.assert_array_equal(
                lr[np.argsort(left_info["players"]["env_id"], kind="stable")],
                rr[np.argsort(right_info["players"]["env_id"], kind="stable")],
            )
            saw_end |= bool(np.any(lt | lx))
        self.assertTrue(saw_end)
        left.close()
        right.close()

    @parameterized.parameters(1, 3, 11)
    def test_soccer_player_batches(self, team: int) -> None:
        """Keep player values separate from match-level episode boundaries."""
        env = make_dm(
            "DmcSoccerBoxhead-v1",
            team_size=team,
            num_envs=2,
            num_threads=2,
            max_episode_steps=4,
        )
        timestep = env.reset()
        self.assertEqual(timestep.reward.shape, (4 * team,))
        self.assertEqual(timestep.step_type.shape, (2,))
        for step in range(9):
            timestep = env.step(np.full((4 * team, 3), 0.1 * np.sin(step)))
            players = timestep.observation.players.env_id
            self.assertEqual(np.count_nonzero(players == 0), 2 * team)
            self.assertEqual(np.count_nonzero(players == 1), 2 * team)
            self.assertEqual(
                timestep.observation.obs["joints_pos"].shape, (4 * team, 1, 1)
            )
            np.testing.assert_array_equal(timestep.last(), [step in (3, 8)] * 2)
        env.close()

    def test_incomplete_soccer_actions(self) -> None:
        """Reject missing or misrouted players before entering native workers."""
        env = make_gymnasium("DmcSoccerBoxhead-v1", num_envs=2, team_size=1)
        env.reset()
        env.step(np.zeros((4, 3)))
        with self.assertRaisesRegex(ValueError, "every player"):
            env.step(np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "every player"):
            env.step({
                "action": np.zeros((4, 3)),
                "players": {"env_id": np.array([0, 0, 0, 1], np.int32)},
            })
        # A rejected batch must leave the pool usable.
        _, reward, _, _, _ = env.step(np.zeros((4, 3)))
        self.assertEqual(reward.shape, (4,))
        env.close()


if __name__ == "__main__":
    absltest.main()
