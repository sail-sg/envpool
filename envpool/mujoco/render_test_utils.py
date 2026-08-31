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

"""Shared RGB comparison for native and oracle render tests."""

import platform

import numpy as np


def assert_rgb_images(
    actual: np.ndarray | None,
    expected: np.ndarray | None,
    context: str = "",
) -> None:
    """Compare RGB frames without requiring identical CGL rounding."""
    assert actual is not None and expected is not None, context
    np.testing.assert_equal(actual.shape, expected.shape)
    np.testing.assert_equal(actual.dtype, np.uint8)
    np.testing.assert_equal(expected.dtype, np.uint8)
    if platform.system() != "Darwin":
        np.testing.assert_array_equal(actual, expected, err_msg=context)
        return
    # Identical scenes can produce sparse CGL/Metal color differences. Bound
    # both their magnitude and mean per frame, without redrawing at runtime
    # or tuning limits for individual tasks, seeds, or image resolutions.
    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    np.testing.assert_array_less(
        delta.max(axis=(-3, -2, -1)), 6, err_msg=context
    )
    np.testing.assert_allclose(
        delta.mean(axis=(-3, -2, -1)), 0, rtol=0, atol=0.01, err_msg=context
    )
