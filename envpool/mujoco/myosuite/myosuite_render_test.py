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
"""Render smoke tests for native MyoSuite envs."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from absl.testing import absltest

from envpool.registration import make_gymnasium

importlib.import_module("envpool.mujoco.myosuite.registration")

_TASK_IDS = (
    "myoFingerReachFixed-v0",
    "myoHandReachFixed-v0",
    "myoLegWalk-v0",
    "myoChallengeDieReorientP1-v0",
    "MyoHandAirplaneFixed-v0",
)
_WIDTH = 64
_HEIGHT = 48
_ORACLE_RENDER_TASK_ID = "myoFingerReachFixed-v0"


def _oracle_probe_path() -> Path:
    runfiles = Path(os.environ["TEST_SRCDIR"])
    workspace = os.environ.get("TEST_WORKSPACE", "envpool")
    candidates = [
        runfiles / workspace / "envpool/mujoco/myosuite_oracle_probe",
        runfiles / workspace / "envpool/mujoco/myosuite_oracle_probe.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = list(runfiles.rglob("myosuite_oracle_probe"))
    if matches:
        return matches[0]
    raise RuntimeError(
        f"could not locate myosuite_oracle_probe under {runfiles}"
    )


def _oracle_trace(task_id: str) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out:
        out_path = Path(out.name)
    cmd = [
        str(_oracle_probe_path()),
        "--mode",
        "trace",
        "--steps",
        "3",
        "--seed",
        "3",
        "--task_id",
        task_id,
        "--out",
        str(out_path),
    ]
    env = os.environ.copy()
    env["ROBOHIVE_VERBOSITY"] = "SILENT"
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            env=env,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "MyoSuite oracle probe failed\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        return json.loads(out_path.read_text())["tasks"][task_id]
    finally:
        out_path.unlink(missing_ok=True)


def _render(env: Any) -> np.ndarray:
    frame = env.render()
    if frame is None:
        raise AssertionError("MyoSuite render returned None")
    return frame


class MyoSuiteRenderTest(absltest.TestCase):
    """Validate native MyoSuite RGB rendering after reset and steps."""

    def test_reset_and_first_three_step_render(self) -> None:
        """Render reset and the first three deterministic zero-action steps."""
        for task_id in _TASK_IDS:
            with self.subTest(task_id=task_id):
                env = make_gymnasium(
                    task_id,
                    num_envs=1,
                    seed=3,
                    render_mode="rgb_array",
                    render_width=_WIDTH,
                    render_height=_HEIGHT,
                )
                try:
                    env.reset()
                    frames = [_render(env)]
                    action = np.zeros(
                        (1, *env.action_space.shape), dtype=np.float32
                    )
                    for _ in range(3):
                        env.step(action)
                        frames.append(_render(env))
                    for frame in frames:
                        self.assertEqual(frame.shape, (1, _HEIGHT, _WIDTH, 3))
                        self.assertEqual(frame.dtype, np.uint8)
                        self.assertGreater(int(frame.max()), int(frame.min()))
                finally:
                    env.close()

    def test_official_trace_native_render_bitwise(self) -> None:
        """Official reset/action trace must render bitwise under native replay."""
        oracle = _oracle_trace(_ORACLE_RENDER_TASK_ID)
        env_a = make_gymnasium(
            _ORACLE_RENDER_TASK_ID,
            num_envs=1,
            seed=3,
            render_mode="rgb_array",
            render_width=_WIDTH,
            render_height=_HEIGHT,
        )
        env_b = make_gymnasium(
            _ORACLE_RENDER_TASK_ID,
            num_envs=1,
            seed=3,
            render_mode="rgb_array",
            render_width=_WIDTH,
            render_height=_HEIGHT,
        )
        try:
            env_a.reset()
            env_b.reset()
            frames_a = [_render(env_a)[0]]
            frames_b = [_render(env_b)[0]]
            for action in oracle["actions"]:
                batched_action = np.asarray(action, dtype=np.float32)[None, :]
                env_a.step(batched_action)
                env_b.step(batched_action)
                frames_a.append(_render(env_a)[0])
                frames_b.append(_render(env_b)[0])
            self.assertLen(frames_a, 4)
            self.assertLen(frames_b, 4)
            for step_id, (frame_a, frame_b) in enumerate(
                zip(frames_a, frames_b, strict=True)
            ):
                np.testing.assert_array_equal(
                    frame_a,
                    frame_b,
                    err_msg=(f"{_ORACLE_RENDER_TASK_ID} render step {step_id}"),
                )
                self.assertGreater(int(frame_a.max()), int(frame_a.min()))
        finally:
            env_a.close()
            env_b.close()


if __name__ == "__main__":
    absltest.main()
