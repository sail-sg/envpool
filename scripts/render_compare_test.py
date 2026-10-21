#!/usr/bin/env python3

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
"""Exercise Playground's documentation renderer against the native images."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import render_compare


class PlaygroundRenderCompareTest(unittest.TestCase):
    """Keep documentation generation on the tested all-task oracle path."""

    def test_all_playground_tasks(self) -> None:
        """Generate the documented comparison image with every registered task."""
        with tempfile.TemporaryDirectory() as temp:
            output = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", temp))
            output /= "mujoco_playground_official_compare.png"
            # Use the documented generation command and its image budgets.
            with mock.patch.object(
                sys,
                "argv",
                [
                    "render_compare",
                    "--family=mujoco_playground",
                    f"--output={output}",
                    "--columns=4",
                    "--source-width=480",
                    "--source-height=360",
                    "--tile-width=144",
                    "--tile-height=108",
                    "--max-mean-abs-diff=4.2",
                    "--max-mismatch-ratio=0.15",
                    "--max-ignored-abs-diff=3",
                ],
            ):
                render_compare.main()
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
