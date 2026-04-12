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
"""Render API tests for EnvPool gfootball."""

from __future__ import annotations

from absl.testing import absltest

from envpool.gfootball.gfootball_oracle_util import (
    ALL_TASK_IDS,
    register_gfootball_envs,
)
from envpool.registration import make_gymnasium

register_gfootball_envs()


class _GfootballRenderTest(absltest.TestCase):
    def assert_task_render_unsupported(self, task_id: str) -> None:
        for render_mode in ("rgb_array", "human"):
            env = make_gymnasium(
                task_id, num_envs=1, seed=0, render_mode=render_mode
            )
            try:
                env.reset()
                with self.assertRaisesRegex(
                    RuntimeError, "render not implemented"
                ):
                    env.render()
            finally:
                env.close()

    def test_all_registered_tasks_reject_render(self) -> None:
        for task_id in ALL_TASK_IDS:
            with self.subTest(task_id=task_id):
                self.assert_task_render_unsupported(task_id)


if __name__ == "__main__":
    absltest.main()
