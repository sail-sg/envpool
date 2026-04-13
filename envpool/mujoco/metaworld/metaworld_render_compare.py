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
"""Generate the MetaWorld EnvPool-vs-official render comparison image."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envpool.python.glfw_context import preload_windows_gl_dlls

preload_windows_gl_dlls(strict=True)

import mujoco  # noqa: E402
from metaworld.env_dict import ALL_V3_ENVIRONMENTS  # noqa: E402

import envpool.mujoco.metaworld.registration as metaworld_registration  # noqa: E402
from envpool.registration import make_gymnasium  # noqa: E402

_TILE_WIDTH = 96
_TILE_HEIGHT = 72
# Render both sides at Gymnasium's default source size, then downsample only for
# the docs mosaic. Rendering source frames directly at thumbnail size changes
# MuJoCo offscreen antialiasing behavior.
_SOURCE_WIDTH = 480
_SOURCE_HEIGHT = 480
_PAIR_GAP = 4
_CELL_WIDTH = 230
_CELL_HEIGHT = 104
_CELL_GAP = 12
_MARGIN = 16
_COLUMNS = 5
_SEED = 7
_CAMERA_ID = 1  # MetaWorld's fixed "corner" camera.
_MAX_MEAN_ABS_DIFF = 0.25
_MAX_MISMATCH_RATIO = 0.005


def _make_oracle(task_name: str) -> Any:
    env = ALL_V3_ENVIRONMENTS[task_name](
        render_mode="rgb_array",
        camera_id=_CAMERA_ID,
    )
    env._set_task_called = True
    env._partially_observable = True
    env.mujoco_renderer.width = _SOURCE_WIDTH
    env.mujoco_renderer.height = _SOURCE_HEIGHT
    return env


def _sync_reset_state(oracle: Any, info: dict[str, Any]) -> None:
    required_keys = (
        "rand_vec0",
        "qpos0",
        "qvel0",
        "mocap_pos0",
        "mocap_quat0",
        "qacc0",
        "qacc_warmstart0",
        "init_tcp0",
        "init_left_pad0",
        "init_right_pad0",
    )
    missing = [key for key in required_keys if key not in info]
    if missing:
        raise RuntimeError(
            "MetaWorld render comparison must be generated with "
            "`bazel run --config=debug //envpool/mujoco:metaworld_render_compare` "
            f"so EnvPool exposes reset-sync info. Missing keys: {missing}"
        )

    rand_vec = np.asarray(info["rand_vec0"][0], dtype=np.float64)
    random_dim = int(oracle._random_reset_space.low.size)
    oracle._freeze_rand_vec = True
    oracle._last_rand_vec = rand_vec[:random_dim].copy()
    oracle.reset()

    qpos = np.asarray(info["qpos0"][0], dtype=np.float64)[
        : oracle.data.qpos.size
    ]
    qvel = np.asarray(info["qvel0"][0], dtype=np.float64)[
        : oracle.data.qvel.size
    ]
    oracle.set_state(qpos, qvel)
    oracle.data.mocap_pos[0] = np.asarray(
        info["mocap_pos0"][0], dtype=np.float64
    )
    oracle.data.mocap_quat[0] = np.asarray(
        info["mocap_quat0"][0], dtype=np.float64
    )
    oracle.data.qacc[:] = np.asarray(info["qacc0"][0], dtype=np.float64)[
        : oracle.data.qacc.size
    ]
    oracle.data.qacc_warmstart[:] = np.asarray(
        info["qacc_warmstart0"][0], dtype=np.float64
    )[: oracle.data.qacc_warmstart.size]
    mujoco.mj_forward(oracle.model, oracle.data)
    oracle.init_tcp = np.asarray(info["init_tcp0"][0], dtype=np.float64).copy()
    oracle.init_left_pad = np.asarray(
        info["init_left_pad0"][0], dtype=np.float64
    ).copy()
    oracle.init_right_pad = np.asarray(
        info["init_right_pad0"][0], dtype=np.float64
    ).copy()
    if hasattr(oracle, "_handle_init_pos"):
        oracle._handle_init_pos = oracle._get_pos_objects().copy()

    curr_obs = oracle._get_curr_obs_combined_no_goal()
    oracle._prev_obs = curr_obs.copy()
    obs = oracle._get_obs().astype(np.float64)
    oracle._last_stable_obs = obs.copy()


def _resize_frame(frame: np.ndarray) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    if image.size == (_TILE_WIDTH, _TILE_HEIGHT):
        return image
    return image.resize((_TILE_WIDTH, _TILE_HEIGHT), Image.Resampling.LANCZOS)


def _assert_frames_bitwise_equal(
    task_name: str, envpool_frame: np.ndarray, official_frame: np.ndarray
) -> None:
    if envpool_frame.shape != official_frame.shape:
        raise RuntimeError(
            f"{task_name} render shape mismatch: "
            f"{envpool_frame.shape} != {official_frame.shape}"
        )
    if np.array_equal(envpool_frame, official_frame):
        return
    diff = np.abs(
        envpool_frame.astype(np.int16) - official_frame.astype(np.int16)
    )
    mismatch_ratio = float(np.count_nonzero(diff)) / float(diff.size)
    raise RuntimeError(
        f"{task_name} render mismatch: "
        f"mean_abs_diff={float(diff.mean()):.6f}, "
        f"max_abs_diff={int(diff.max())}, "
        f"mismatch_ratio={mismatch_ratio:.6%}"
    )


def _assert_frames_close(
    task_name: str, envpool_frame: np.ndarray, official_frame: np.ndarray
) -> None:
    if envpool_frame.shape != official_frame.shape:
        raise RuntimeError(
            f"{task_name} render shape mismatch: "
            f"{envpool_frame.shape} != {official_frame.shape}"
        )
    diff = np.abs(
        envpool_frame.astype(np.int16) - official_frame.astype(np.int16)
    )
    if diff.size == 0:
        return
    mismatch_ratio = float(np.count_nonzero(diff)) / float(diff.size)
    mean_abs_diff = float(diff.mean())
    if (
        mean_abs_diff > _MAX_MEAN_ABS_DIFF
        or mismatch_ratio > _MAX_MISMATCH_RATIO
    ):
        raise RuntimeError(
            f"{task_name} render mismatch: "
            f"mean_abs_diff={mean_abs_diff:.6f}, "
            f"max_abs_diff={int(diff.max())}, "
            f"mismatch_ratio={mismatch_ratio:.6%}"
        )


def _render_pair(
    task_name: str, *, require_bitwise: bool
) -> tuple[Image.Image, Image.Image]:
    task_id = f"Meta-World/{task_name}"
    env = make_gymnasium(
        task_id,
        num_envs=1,
        seed=_SEED,
        render_mode="rgb_array",
        render_width=_SOURCE_WIDTH,
        render_height=_SOURCE_HEIGHT,
        render_camera_id=_CAMERA_ID,
    )
    oracle = _make_oracle(task_name)
    try:
        _, info = env.reset()
        _sync_reset_state(oracle, info)
        envpool_frame = env.render()
        assert envpool_frame is not None
        official_frame = oracle.render()
        if require_bitwise:
            _assert_frames_bitwise_equal(
                task_name, envpool_frame[0], official_frame
            )
        else:
            _assert_frames_close(task_name, envpool_frame[0], official_frame)
        return _resize_frame(envpool_frame[0]), _resize_frame(official_frame)
    finally:
        env.close()
        oracle.close()


def _draw_panel(
    canvas: Image.Image,
    task_name: str,
    envpool_image: Image.Image,
    official_image: Image.Image,
    index: int,
    font: Any,
) -> None:
    draw = ImageDraw.Draw(canvas)
    col = index % _COLUMNS
    row = index // _COLUMNS
    left = _MARGIN + col * (_CELL_WIDTH + _CELL_GAP)
    top = _MARGIN + row * (_CELL_HEIGHT + _CELL_GAP)
    frame_top = top + 28
    official_left = left + _TILE_WIDTH + _PAIR_GAP

    draw.text((left, top), task_name, fill=(30, 30, 30), font=font)
    draw.text((left, top + 14), "EnvPool", fill=(80, 80, 80), font=font)
    draw.text(
        (official_left, top + 14), "Official", fill=(80, 80, 80), font=font
    )
    canvas.paste(envpool_image, (left, frame_top))
    canvas.paste(official_image, (official_left, frame_top))
    draw.rectangle(
        [left, frame_top, left + _TILE_WIDTH - 1, frame_top + _TILE_HEIGHT - 1],
        outline=(205, 205, 205),
    )
    draw.rectangle(
        [
            official_left,
            frame_top,
            official_left + _TILE_WIDTH - 1,
            frame_top + _TILE_HEIGHT - 1,
        ],
        outline=(205, 205, 205),
    )


def generate(output: Path, *, require_bitwise: bool) -> None:
    """Generate and write the full MetaWorld render comparison image."""
    task_names = tuple(metaworld_registration.metaworld_v3_envs)
    rows = math.ceil(len(task_names) / _COLUMNS)
    width = _MARGIN * 2 + _COLUMNS * _CELL_WIDTH + (_COLUMNS - 1) * _CELL_GAP
    height = _MARGIN * 2 + rows * _CELL_HEIGHT + (rows - 1) * _CELL_GAP
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    font = ImageFont.load_default()

    for index, task_name in enumerate(task_names):
        envpool_image, official_image = _render_pair(
            task_name,
            require_bitwise=require_bitwise,
        )
        _draw_panel(
            canvas,
            task_name,
            envpool_image,
            official_image,
            index,
            font,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    """Parse command-line arguments and generate the comparison image."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/_static/render_samples/metaworld_official_compare.png"
        ),
    )
    parser.add_argument("--require-bitwise", action="store_true")
    args = parser.parse_args()
    output = args.output
    if not output.is_absolute():
        output = Path(
            os.environ.get("BUILD_WORKSPACE_DIRECTORY", ".")
        ).joinpath(output)
    generate(output, require_bitwise=args.require_bitwise)


if __name__ == "__main__":
    main()
