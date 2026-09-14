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

import os
import platform
import tempfile

import numpy as np


def assert_rgb_images(
    actual: np.ndarray | None,
    expected: np.ndarray | None,
    context: str = "",
    *,
    macos_peak_error: int = 5,
    macos_mean_error: float = 0.01,
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
    # both their magnitude and mean per frame. Camera-specific exceptions
    # must document their independently reproduced renderer error.
    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    try:
        np.testing.assert_array_less(
            delta.max(axis=(-3, -2, -1)), macos_peak_error + 1, err_msg=context
        )
        np.testing.assert_allclose(
            delta.mean(axis=(-3, -2, -1)),
            0,
            rtol=0,
            atol=macos_mean_error,
            err_msg=context,
        )
    except AssertionError:
        print(
            f"{context}: RGB error sum={delta.sum(axis=(-3, -2, -1))}, "
            f"changed pixels={np.count_nonzero(np.any(delta, axis=-1), axis=(-2, -1))}"
        )
        output_dir = os.environ.get(
            "TEST_UNDECLARED_OUTPUTS_DIR"
        ) or os.environ.get("ENVPOOL_TEST_OUTPUT_DIR")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=output_dir,
                prefix="render-mismatch-",
                suffix=".npz",
                delete=False,
            ) as output:
                np.savez_compressed(
                    output, actual=actual, expected=expected, context=context
                )
            print(f"Saved RGB mismatch: {output.name}")
        raise
