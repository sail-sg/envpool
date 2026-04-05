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
"""Tests for native DMC pixel observations."""

from absl.testing import absltest

import envpool.mujoco.dmc.registration  # noqa: F401
from envpool.mujoco.pixel_observation_test_utils import (
    assert_frame_stack_rolls_in_channel_dimension,
    assert_make_spec_exposes_bchw_pixel_specs,
    assert_tasks_align_with_render_for_three_steps,
)

_IMPORT_PATH = "envpool.mujoco.dmc"


class MujocoDMCPixelObservationTest(absltest.TestCase):
    """Pixel-observation tests for DMC tasks."""

    def test_make_spec_exposes_bchw_pixel_specs(self) -> None:
        """`make_spec` should expose channel-first pixel observations."""
        assert_make_spec_exposes_bchw_pixel_specs(self, _IMPORT_PATH)

    def test_frame_stack_rolls_in_channel_dimension(self) -> None:
        """Frame stacking should shift along the channel dimension."""
        assert_frame_stack_rolls_in_channel_dimension(self, _IMPORT_PATH)

    def test_all_dmc_tasks_align_with_render_for_three_steps(self) -> None:
        """All DMC tasks should match `render()` for reset + 3 steps."""
        assert_tasks_align_with_render_for_three_steps(self, _IMPORT_PATH)


if __name__ == "__main__":
    absltest.main()
