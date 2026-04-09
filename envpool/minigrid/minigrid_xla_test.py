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

"""XLA smoke tests for the MiniGrid backend."""

import sys
from typing import Any

import jax
import jax.numpy as jnp
from absl.testing import absltest

from envpool.minigrid import MiniGridEnvSpec, MiniGridGymnasiumEnvPool


def _make_empty_5x5(num_envs: int) -> Any:
    config = MiniGridEnvSpec.gen_config(
        env_name="empty",
        max_episode_steps=100,
        num_envs=num_envs,
        size=5,
    )
    return MiniGridGymnasiumEnvPool(MiniGridEnvSpec(config))


class _MiniGridXlaTest(absltest.TestCase):
    def test_jitted_step_without_explicit_reset(self) -> None:
        env = _make_empty_5x5(num_envs=8)
        if sys.platform in ("darwin", "win32"):
            with self.assertRaisesRegex(RuntimeError, "XLA.*unavailable"):
                env.xla()
            return
        handle, _, _, step = env.xla()

        @jax.jit
        def raw_step(handle: jnp.ndarray, actions: jnp.ndarray) -> Any:
            return step(handle, actions)

        actions = jnp.zeros((8,), dtype=jnp.int32)
        handle, states = raw_step(handle, actions)
        jax.block_until_ready(states)

        obs, reward, terminated, truncated, info = states
        self.assertEqual(obs["image"].shape, (8, 7, 7, 3))
        self.assertEqual(reward.shape, (8,))
        self.assertEqual(terminated.shape, (8,))
        self.assertEqual(truncated.shape, (8,))
        self.assertEqual(info["env_id"].shape, (8,))
        self.assertEqual(handle.shape, (8,))


if __name__ == "__main__":
    absltest.main()
