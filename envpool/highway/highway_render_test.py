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
"""Render tests for Highway environments."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from absl.testing import absltest

import envpool.highway.registration  # noqa: F401
from envpool.highway.highway_align_test import (
    _debug_state,
    _make_oracle,
    _patch_oracle,
    _straight_road_config,
)
from envpool.registration import make_gymnasium

_RENDER_MATCH_ACTIONS = (1, 3, 3, 4)
_RENDER_MATCH_CASES = (
    (
        "highway_v0_no_traffic",
        "Highway-v0",
        "highway-v0",
        {"vehicles_count": 0, "initial_lane_id": 1},
        _RENDER_MATCH_ACTIONS,
    ),
    (
        "highway_fast_v0_no_traffic",
        "HighwayFast-v0",
        "highway-fast-v0",
        {
            "vehicles_count": 0,
            "initial_lane_id": 1,
            "duration": 30,
            "simulation_frequency": 5,
            "lanes_count": 3,
        },
        _RENDER_MATCH_ACTIONS,
    ),
    (
        "highway_v0_two_lane_no_traffic",
        "Highway-v0",
        "highway-v0",
        {"vehicles_count": 0, "lanes_count": 2, "initial_lane_id": 0},
        _RENDER_MATCH_ACTIONS,
    ),
    (
        "highway_v0_small_traffic",
        "Highway-v0",
        "highway-v0",
        {"vehicles_count": 6, "lanes_count": 3, "initial_lane_id": 1},
        (),
    ),
)


def _render_array(env: Any, env_ids: Any = None) -> np.ndarray:
    frame = env.render(env_ids=env_ids)
    assert frame is not None
    return cast(np.ndarray, frame)


def _zero_action(space: Any, num_envs: int) -> np.ndarray:
    return np.full((num_envs,), 1, dtype=np.int64)


class _HighwayRenderTest(absltest.TestCase):
    def test_render_is_batch_consistent_and_state_invariant(self) -> None:
        env = make_gymnasium(
            "Highway-v0",
            num_envs=2,
            seed=0,
            render_mode="rgb_array",
            render_width=120,
            render_height=64,
        )
        try:
            env.reset()
            for step in range(3):
                frame0 = _render_array(env)
                frame1 = _render_array(env, env_ids=1)
                frames = _render_array(env, env_ids=[0, 1])
                frame0_again = _render_array(env)

                self.assertEqual(frame0.shape, (1, 64, 120, 3))
                self.assertEqual(frame1.shape, (1, 64, 120, 3))
                self.assertEqual(frames.shape, (2, 64, 120, 3))
                self.assertEqual(frame0.dtype, np.uint8)
                np.testing.assert_array_equal(frame0[0], frames[0])
                np.testing.assert_array_equal(frame1[0], frames[1])
                np.testing.assert_array_equal(frame0, frame0_again)
                self.assertGreater(int(frame0.max()) - int(frame0.min()), 0)

                if step < 2:
                    env.step(_zero_action(env.action_space, 2))
        finally:
            env.close()

    def assert_render_matches_upstream_bitwise(
        self,
        task_id: str,
        oracle_env_id: str,
        config: dict[str, Any],
        actions: tuple[int, ...],
    ) -> None:
        env = make_gymnasium(
            task_id,
            num_envs=1,
            seed=0,
            render_mode="rgb_array",
            **config,
        )
        oracle = _make_oracle(oracle_env_id, _straight_road_config(**config))
        try:
            env.reset()
            oracle.reset(seed=0)

            for action in (None, *actions):
                if action is not None:
                    env.step(np.asarray([action], dtype=np.int64))
                _patch_oracle(oracle, _debug_state(env))

                frame = _render_array(env)[0]
                expected = np.asarray(oracle.render(), dtype=np.uint8)
                self.assertEqual(frame.shape, expected.shape)
                np.testing.assert_array_equal(frame, expected)
        finally:
            env.close()
            oracle.close()

    def test_multi_step_render_matches_upstream_bitwise(self) -> None:
        for (
            name,
            task_id,
            oracle_env_id,
            overrides,
            actions,
        ) in _RENDER_MATCH_CASES:
            with self.subTest(name=name):
                self.assert_render_matches_upstream_bitwise(
                    task_id,
                    oracle_env_id,
                    _straight_road_config(**overrides),
                    actions,
                )


if __name__ == "__main__":
    absltest.main()
