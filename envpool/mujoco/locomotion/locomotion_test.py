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

import platform

import numpy as np
from absl.testing import absltest, parameterized
from envpool.mujoco.locomotion.locomotion_envpool import TASKS

import envpool.mujoco.locomotion.registration  # noqa: F401
from envpool.registration import make_dm, make_gymnasium


def assert_egocentric(
    actual: np.ndarray, expected: np.ndarray, task: str, context: str
) -> None:
    """Check camera pixels, including the isolated CGL rounding residual."""
    if platform.system() == "Darwin" and task in {
        "cmu_humanoid_maze_forage",
        "cmu_humanoid_heterogeneous_forage",
        "rodent_maze_forage",
    }:
        # CGL can change color levels even when the official Camera renders
        # the very same MjvScene twice: reproduced over 10,000 frames with
        # bitwise-equal skin vertices/normals and copied MuJoCo state.
        # The CMU cameras also exercise context migration across workers:
        # up to five channels differ by at most two levels; the rodent camera
        # needs only one channel/level. Keep these budgets within each 64x64
        # frame. Other task images, public renders and physics stay bitwise.
        np.testing.assert_equal(actual.shape, expected.shape)
        np.testing.assert_equal(actual.dtype, expected.dtype)
        delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        cmu = task.startswith("cmu_")
        np.testing.assert_array_less(delta, 3 if cmu else 2, err_msg=context)
        np.testing.assert_array_less(
            np.count_nonzero(delta, axis=(-3, -2, -1)),
            6 if cmu else 2,
            err_msg=context,
        )
    else:
        np.testing.assert_array_equal(actual, expected, err_msg=context)


class LocomotionTest(parameterized.TestCase):
    """Check public batching, rendering, and repeatable complete rollouts."""

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
            self.assertEqual(left_obs.keys(), right_obs.keys())
            for key in left_obs:
                context = f"{task}: step {step}, {key}"
                if key == "walker/egocentric_camera":
                    assert_egocentric(
                        left_obs[key][li], right_obs[key][ri], task, context
                    )
                else:
                    np.testing.assert_array_equal(
                        left_obs[key][li], right_obs[key][ri], err_msg=context
                    )
                self.assertTrue(
                    left.observation_space[key].contains(left_obs[key][0]), key
                )
            if step in (0, 32, 97, 194):
                a = left.render(env_ids=[1, 0])
                b = right.render(env_ids=[1, 0])
                assert a is not None and b is not None
                self.assertEqual(a.shape, (2, 80, 96, 3))
                self.assertEqual(a.dtype, np.uint8)
                np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(a[:1], left.render(env_ids=[1]))
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
