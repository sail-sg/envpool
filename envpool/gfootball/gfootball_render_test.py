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
"""Render alignment tests for EnvPool gfootball."""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

from envpool.gfootball.gfootball_oracle_util import (
    ALL_TASK_IDS,
    GfootballOracle,
    register_gfootball_envs,
)
from envpool.registration import make_gymnasium

register_gfootball_envs()

_RENDER_ACTIONS = (5, 11, 13)


def _scalar(value: np.ndarray) -> int:
    return int(np.asarray(value).reshape(-1)[0])


class _GfootballRenderTest(absltest.TestCase):
    def assert_task_render_aligned(self, task_id: str) -> None:
        env = make_gymnasium(
            task_id, num_envs=1, seed=0, render_mode="rgb_array"
        )
        oracle = GfootballOracle(task_id, render=True)
        try:
            _, info = env.reset()
            oracle.reset(
                engine_seed=_scalar(info["engine_seed"]),
                episode_number=_scalar(info["episode_number"]),
            )
            for action in (None, *_RENDER_ACTIONS):
                frame = env.render()
                assert frame is not None
                np.testing.assert_array_equal(frame[0], oracle.render())
                if action is None:
                    continue
                _, _, term, trunc, _ = env.step(
                    np.asarray([action], dtype=np.int32)
                )
                oracle.step(action)
                if bool(term[0] or trunc[0]):
                    break
        finally:
            env.close()

    def test_all_registered_tasks_render_bitwise(self) -> None:
        for task_id in ALL_TASK_IDS:
            with self.subTest(task_id=task_id):
                self.assert_task_render_aligned(task_id)


if __name__ == "__main__":
    absltest.main()
