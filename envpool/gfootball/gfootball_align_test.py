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
"""Alignment tests for EnvPool gfootball."""

from __future__ import annotations

from typing import Any

import numpy as np
from absl.testing import absltest

from envpool.gfootball.gfootball_oracle_util import (
    ALL_TASK_IDS,
    GfootballOracle,
    register_gfootball_envs,
)
from envpool.registration import make_gymnasium

register_gfootball_envs()

_ALIGN_ACTIONS = (5, 5, 11, 5, 13, 5, 15, 3, 7, 12, 0, 17, 18)


def _scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    return arr.reshape(-1)[0].item()


def _assert_info_matches(
    test_case: absltest.TestCase,
    info: dict[str, np.ndarray],
    expected: dict[str, Any],
) -> None:
    for key in (
        "score",
        "game_mode",
        "ball_owned_team",
        "ball_owned_player",
        "steps_left",
        "engine_seed",
        "episode_number",
        "elapsed_step",
    ):
        actual = np.asarray(info[key])[0]
        np.testing.assert_array_equal(actual, expected[key], err_msg=key)


class _GfootballAlignTest(absltest.TestCase):
    def assert_task_aligned(self, task_id: str) -> None:
        env = make_gymnasium(task_id, num_envs=1, seed=0)
        oracle = GfootballOracle(task_id)
        try:
            obs, info = env.reset()
            oracle_obs, oracle_info = oracle.reset(
                engine_seed=int(_scalar(info["engine_seed"])),
                episode_number=int(_scalar(info["episode_number"])),
            )
            np.testing.assert_array_equal(obs[0], oracle_obs)
            _assert_info_matches(self, info, oracle_info)

            for action in _ALIGN_ACTIONS:
                obs, rew, term, trunc, info = env.step(
                    np.asarray([action], dtype=np.int32)
                )
                (
                    oracle_obs,
                    oracle_rew,
                    oracle_term,
                    oracle_trunc,
                    oracle_info,
                ) = oracle.step(action)
                np.testing.assert_array_equal(obs[0], oracle_obs)
                np.testing.assert_array_equal(rew, np.asarray([oracle_rew]))
                np.testing.assert_array_equal(
                    term, np.asarray([oracle_term], dtype=np.bool_)
                )
                np.testing.assert_array_equal(
                    trunc, np.asarray([oracle_trunc], dtype=np.bool_)
                )
                _assert_info_matches(self, info, oracle_info)
                if bool(term[0] or trunc[0]):
                    break
        finally:
            env.close()

    def test_all_registered_tasks_align_bitwise(self) -> None:
        for task_id in ALL_TASK_IDS:
            with self.subTest(task_id=task_id):
                self.assert_task_aligned(task_id)


if __name__ == "__main__":
    absltest.main()
