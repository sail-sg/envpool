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
"""Shared behavioral test inputs, also usable against installed wheels."""

import os
from pathlib import Path
from typing import Any

import numpy as np


def motion_file() -> str:
    """Locate the motion made by the pinned official CSV-to-NPZ converter."""
    if path := os.environ.get("ENVPOOL_MJLAB_TEST_MOTION"):
        return str(Path(path).resolve(strict=True))
    suffix = "third_party/mjlab/generated/testdata/motion.npz"
    if manifest := os.environ.get("RUNFILES_MANIFEST_FILE"):
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            logical, _, physical = line.partition(" ")
            if logical.replace("\\", "/").endswith(suffix):
                return str(Path(physical or logical).resolve(strict=True))
    roots = [Path(os.environ.get("TEST_SRCDIR", "."))]
    roots.extend(
        p
        for p in Path(__file__).absolute().parents
        if p.name.endswith(".runfiles")
    )
    for root in roots:
        for workspace in (
            os.environ.get("TEST_WORKSPACE", "_main"),
            "_main",
            "envpool",
        ):
            path = root / workspace / suffix
            if path.is_file():
                return str(path.resolve())
    raise FileNotFoundError(
        "set ENVPOOL_MJLAB_TEST_MOTION to the generated test motion.npz"
    )


def task_options(task: str) -> dict[str, Any]:
    """Provide the required motion input without changing any task defaults."""
    return {"motion_file": motion_file()} if "Tracking" in task else {}


def actions(size: int, steps: int, seed: int = 31) -> np.ndarray:
    """Mix smooth, random, zero and opposing controls without an oracle policy."""
    rng = np.random.default_rng(seed)
    result = rng.uniform(-0.3, 0.3, (steps, size)).astype(np.float32)
    phase = np.arange(size, dtype=np.float32)
    for step in range(steps):
        if step % 83 < 23:
            result[step] = 0.15 * np.sin(phase * 0.3 + step * 0.17)
        elif step % 83 < 31:
            result[step] = 0
        elif step % 83 > 70:
            result[step] = -result[step - 1]
    return result


def assert_observations(
    actual: dict, expected: dict, context: str = ""
) -> None:
    """Compare every observation group, including visual channels, exactly."""
    np.testing.assert_equal(sorted(actual), sorted(expected))
    for name in actual:
        np.testing.assert_array_equal(
            actual[name], expected[name], err_msg=f"{context}: {name}"
        )


def public_components(task: str, obs: dict[str, np.ndarray], slot: int) -> dict:
    """Use noise-free semantic observations, never counters or render noise."""
    critic = obs["critic"][slot]
    if "Cartpole" in task:
        return {"cart": critic[[0, 3]], "pole": critic[[1, 2, 4]]}
    if "Velocity" in task:
        joints = 12 if "Go1" in task else 29
        offset = 9 + 3 * joints
        # Upstream keeps joint positions/velocities fixed. Root x/y/yaw are
        # intentionally absent from these observations; native-state tests
        # check those independently instead of mistaking sensor noise for DR.
        return {"goal": critic[offset : offset + 3]}
    if "Tracking" in task:
        return {"motion": critic[:58], "pose": critic[67:109]}
    # The Yam robot's reset is fixed; cubes and target positions are random.
    if "camera" in obs:
        return {"objects": obs["camera"][slot], "goal": obs["actor"][slot, -3:]}
    # Both distances use the fixed robot frame. Adding them removes the
    # randomized object position, exposing the goal relative to the fixed
    # reset gripper. Object motion must not conceal a frozen target (#432).
    return {"object": critic[16:19], "goal": critic[16:19] + critic[19:22]}
