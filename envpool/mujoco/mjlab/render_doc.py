# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Make documentation comparisons from the complete-episode alignment harness."""

import argparse
import os
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
from absl import flags

from envpool.mujoco.mjlab import TASKS
from envpool.mujoco.mjlab.mjlab_align_test import MjlabAlignTest, _terrain_cases

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def generate(output: Path) -> None:
    """Plot native-left / official-right frames after verified action rollouts."""
    output.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {}
    case = MjlabAlignTest("test_registry")
    # The final column in the pinned presets is waves. A one-slot rough pool
    # starts on the flat column, which would hide terrain in these examples.
    terrain = {
        task: (count, column) for _, task, count, column in _terrain_cases()
    }
    for task in TASKS:
        native, official = case.compare_episode(
            task, 17, False, *terrain.get(task, (1, 0)), render_size=(320, 240)
        )
        frame = min(3, len(native) - 1)
        group = (
            "cartpole"
            if "Cartpole" in task
            else "velocity"
            if "Velocity" in task
            else "tracking"
            if "Tracking" in task
            else "manipulation"
        )
        groups.setdefault(group, []).append((
            task,
            native[frame],
            official[frame],
        ))
        print(f"{task}: complete episode and rendered frames match", flush=True)
    for group, rows in groups.items():
        fig, axes = plt.subplots(
            len(rows),
            2,
            figsize=(8, 3.2 * len(rows)),
            squeeze=False,
            layout="constrained",
        )
        for row, (task, native, official) in enumerate(rows):
            for column, frame in enumerate((native, official)):
                axes[row, column].imshow(frame, interpolation="none")
                axes[row, column].set_axis_off()
                label = "EnvPool" if column == 0 else "MJLab 1.6.0"
                axes[row, column].set_title(
                    f"{label}\n{task.removeprefix('Mjlab-')}", fontsize=9
                )
        fig.savefig(output / f"mjlab-{group}.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # The reused test harness needs an isolated scratch directory, not a source
    # checkout import or an installed copy of the oracle's Python environment.
    flags.FLAGS(["render_doc"])
    with tempfile.TemporaryDirectory(prefix="mjlab-doc-") as scratch:
        os.environ.setdefault("TEST_SRCDIR", os.environ["RUNFILES_DIR"])
        os.environ["TEST_TMPDIR"] = scratch
        flags.FLAGS.test_tmpdir = scratch
        generate(args.output)
