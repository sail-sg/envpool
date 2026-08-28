# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exercise registration, batching, deterministic replay, and episode boundaries."""

from typing import Any

import numpy as np
from absl.testing import absltest, parameterized

import envpool.craftax.registration  # noqa: F401
from envpool.craftax._registry import CRAFTAX_IDS
from envpool.python.seed_test_utils import check_seeded_resets
from envpool.registration import list_all_envs, make_dm, make_gymnasium


def assert_tree_equal(left: Any, right: Any) -> None:
    """Compare a complete nested batch without ignoring diagnostic fields."""
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, tuple):
        for a, b in zip(left, right, strict=True):
            assert_tree_equal(a, b)
    else:
        np.testing.assert_array_equal(left, right)


class CraftaxTest(parameterized.TestCase):
    """Exercise the public pool with multiple environments and worker counts."""

    @parameterized.parameters(*CRAFTAX_IDS)
    def test_replay_and_batched_render(self, task_id: str) -> None:
        """Replay actions across thread counts and check selected render order."""
        check_seeded_resets(self, task_id)
        classic = "-Classic-" in task_id
        pixels = "-Pixels-" in task_id
        kwargs = dict(
            num_envs=3, seed=123, max_episode_steps=97, render_mode="rgb_array"
        )
        left = make_gymnasium(task_id, num_threads=1, **kwargs)
        alias = "Craftax/" + task_id.removeprefix("Craftax-")
        self.assertIn(alias, list_all_envs())
        right = make_gymnasium(alias, num_threads=3, **kwargs)
        try:
            a, b = left.reset(), right.reset()
            assert_tree_equal(a, b)
            shape = (63, 63, 3) if classic else (130, 110, 3)
            if not pixels:
                shape = (1345 if classic else 8268,)
            self.assertEqual(a[0].shape, (3, *shape))
            self.assertEqual(a[0].dtype, np.float32)
            self.assertEqual(left.action_space.n, 17 if classic else 43)
            self.assertTrue(left.observation_space.contains(a[0][0]))
            expected_shape = (144, 144, 3) if classic else (208, 176, 3)
            terminated = 0
            rng = np.random.default_rng(23)
            for t in range(400):
                actions = rng.integers(
                    left.action_space.n, size=3, dtype=np.int32
                )
                a, b = left.step(actions), right.step(actions)
                assert_tree_equal(a, b)
                terminated += int(np.count_nonzero(a[2] | a[3]))
                if t % 17 == 0:
                    frame = left.render(env_ids=[2, 0])
                    other = right.render(env_ids=[0, 2])
                    assert frame is not None and other is not None
                    self.assertEqual(frame.shape, (2, *expected_shape))
                    self.assertEqual(frame.dtype, np.uint8)
                    np.testing.assert_array_equal(frame, other[::-1])
                    np.testing.assert_array_equal(
                        frame[1], np.asarray(left.render(env_ids=0))[0]
                    )
            self.assertGreaterEqual(terminated, 9)
        finally:
            left.close()
            right.close()

    @parameterized.parameters(*CRAFTAX_IDS)
    def test_time_limit_and_reset_timing(self, task_id: str) -> None:
        """Check terminal discounts and same-step versus next-step resets."""
        auto = "-AutoReset-" in task_id
        env = make_gymnasium(task_id, num_envs=1, seed=5, max_episode_steps=32)
        dm = make_dm(task_id, num_envs=1, seed=5, max_episode_steps=32)
        try:
            obs, _ = env.reset()
            first = dm.reset()
            self.assertTrue(first.first()[0])
            self.assertEqual(dm.action_spec().num_values, env.action_space.n)
            for t in range(33):
                obs, reward, terminated, truncated, info = env.step(
                    np.array([0], np.int32)
                )
                step = dm.step(np.array([0], np.int32))
                np.testing.assert_array_equal(obs, step.observation.obs)
                np.testing.assert_array_equal(reward, step.reward)
                np.testing.assert_array_equal(info["discount"], step.discount)
                self.assertFalse(terminated[0])
                self.assertEqual(bool(truncated[0]), t == 31)
                self.assertEqual(bool(step.last()[0]), t == 31)
                if t == 32:
                    self.assertEqual(bool(step.first()[0]), not auto)
                    self.assertEqual(info["elapsed_step"][0], 1 if auto else 0)
        finally:
            env.close()
            dm.close()

    @parameterized.parameters(
        "Craftax-Classic-Symbolic-v1", "Craftax-Symbolic-v1"
    )
    def test_resized_render(self, task_id: str) -> None:
        """Resize actual multi-step frames with the documented nearest sampling."""
        kwargs = dict(num_envs=1, seed=9, render_mode="rgb_array")
        env = make_gymnasium(task_id, **kwargs)
        resized = make_gymnasium(
            task_id, render_width=211, render_height=173, **kwargs
        )
        try:
            assert_tree_equal(env.reset(), resized.reset())
            rng = np.random.default_rng(7)
            for t in range(64):
                actions = rng.integers(
                    env.action_space.n, size=1, dtype=np.int32
                )
                assert_tree_equal(env.step(actions), resized.step(actions))
                if t % 13 == 0:
                    frame = np.asarray(env.render())[0]
                    y = np.arange(173) * frame.shape[0] // 173
                    x = np.arange(211) * frame.shape[1] // 211
                    np.testing.assert_array_equal(
                        np.asarray(resized.render())[0], frame[y[:, None], x]
                    )
        finally:
            env.close()
            resized.close()

    @parameterized.parameters(
        "Craftax-Classic-Symbolic-v1", "Craftax-Symbolic-v1"
    )
    def test_invalid_initial_state(self, task_id: str) -> None:
        """Reject malformed saved state before native worker threads can use it."""
        env = make_gymnasium(task_id, num_envs=1, debug_state=True)
        try:
            _, info = env.reset()
            initial = info["state"][0].copy()
        finally:
            env.close()
        for value in (np.nan, 999):
            invalid = initial.copy()
            invalid[0] = value
            with self.assertRaises(ValueError):
                make_gymnasium(
                    task_id, num_envs=1, initial_state=invalid.tolist()
                )
        with self.assertRaises(ValueError):
            make_gymnasium(
                task_id, num_envs=1, initial_state=initial[:-1].tolist()
            )


if __name__ == "__main__":
    absltest.main()
