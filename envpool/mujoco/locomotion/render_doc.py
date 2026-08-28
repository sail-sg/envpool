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
"""Generate native-left / official-right documentation images for every task."""

import argparse
import os
import platform
from pathlib import Path
from typing import Any

from envpool.mujoco.oracle import configure_mujoco_package_shared_lib
from envpool.python.glfw_context import preload_windows_gl_dlls

configure_mujoco_package_shared_lib()
preload_windows_gl_dlls(strict=True)
if platform.system() == "Linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
from envpool.mujoco.locomotion.locomotion_envpool import TASKS
from PIL import Image, ImageDraw, ImageFont

import envpool.mujoco.locomotion.registration  # noqa: F401
from envpool.mujoco.dmc.render_oracle import configure_macos_dm_control_renderer
from envpool.mujoco.locomotion.oracle import make_oracle
from envpool.registration import make_dm

configure_macos_dm_control_renderer()


def generate(output: Path) -> None:
    """Render four external action steps after one reset synchronization."""
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=16)
    groups: dict[str, list[Image.Image]] = {}
    for task in TASKS:
        official = make_oracle(task, seed=0)
        official.reset()
        camera = -1
        if "maze" not in task and "forage" not in task:
            for candidate in ("walker/side", "top_down"):
                camera = (
                    official.physics.model.name2id(candidate, "camera")
                    if candidate
                    in official.physics.named.model.cam_pos.axes.row.names
                    else -1
                )
                if camera >= 0:
                    break
        env: Any = make_dm(
            f"dm_control/locomotion/{task}",
            seed=0,
            render_mode="rgb_array",
            render_width=320,
            render_height=240,
            render_camera_id=camera,
        )
        env.reset()
        state = env._snapshot()
        with official.physics.reset_context():
            official.physics.data.qpos[:] = state["qpos"]
            official.physics.data.qvel[:] = state["qvel"]
            official.physics.data.act[:] = state["act"]
        official.physics.data.qacc_warmstart[:] = state["warmstart"]
        players = 4 if task.startswith("soccer_") else 1
        for step in range(4):
            action = np.full(
                (players, *env.action_spec().shape), 0.05 * np.sin(step)
            )
            if platform.system() == "Darwin":
                official.physics.contexts.gl._platform_make_current()
            official.step(action if players > 1 else action[0])
            env.step(action)
        native_frame = env.render()[0]
        if platform.system() == "Darwin":
            official.physics.contexts.gl._platform_make_current()
        oracle_frame = official.physics.render(240, 320, camera)
        np.testing.assert_array_equal(native_frame, oracle_frame, err_msg=task)
        row = Image.new("RGB", (640, 268), "white")
        ImageDraw.Draw(row).text((8, 5), task, font=font, fill="#263238")
        row.paste(Image.fromarray(native_frame), (0, 28))
        row.paste(Image.fromarray(oracle_frame), (320, 28))
        group = (
            "soccer"
            if task.startswith("soccer_")
            else "rodent"
            if task.startswith("rodent_")
            else "humanoid"
        )
        groups.setdefault(group, []).append(row)
        print(f"{task}: native and official frames match", flush=True)
        env.close()
        official.close()
    for group, rows in groups.items():
        canvas = Image.new("RGB", (640, 28 + 268 * len(rows)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), "EnvPool", font=font, fill="#263238")
        draw.text((328, 5), "dm_control 1.0.44", font=font, fill="#263238")
        for index, row in enumerate(rows):
            canvas.paste(row, (0, 28 + 268 * index))
        canvas.save(output / f"dmc-locomotion-{group}.png", optimize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    generate(parser.parse_args().output)
