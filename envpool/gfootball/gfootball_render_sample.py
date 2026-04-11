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
"""Generate the gfootball docs render comparison image."""

from __future__ import annotations

import argparse
import gc
import math
import os
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envpool.gfootball.gfootball_oracle_util import (
    ALL_TASK_IDS,
    GfootballOracle,
    register_gfootball_envs,
)
from envpool.registration import make_gymnasium

if os.name != "nt" and os.environ.get("ENVPOOL_KEEP_DISPLAY") != "1":
    os.environ.pop("DISPLAY", None)

_GRID_COLUMNS = 3
_FRAME_WIDTH = 128
_FRAME_HEIGHT = 72
_CAPTURE_WIDTH = 320
_CAPTURE_HEIGHT = 180
_DISPLAY_ACTIONS = (5, 11, 13)
_PANEL_PADDING = 10
_IMAGE_GAP = 8
_PANEL_GAP_X = 16
_PANEL_GAP_Y = 18
_OUTER_MARGIN = 14
_HEADER_HEIGHT = 34
_BACKGROUND = (248, 249, 251)
_PANEL_BACKGROUND = (255, 255, 255)
_PANEL_BORDER = (210, 214, 220)
_TEXT = (32, 36, 41)
_SUBTEXT = (88, 95, 105)


def _scalar(value: np.ndarray) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _capture_frames(task_id: str) -> tuple[np.ndarray, np.ndarray]:
    env = make_gymnasium(
        task_id,
        num_envs=1,
        seed=0,
        render_mode="rgb_array",
        render_resolution_x=_CAPTURE_WIDTH,
        render_resolution_y=_CAPTURE_HEIGHT,
    )
    try:
        _, info = env.reset()
        frame = env.render()
        assert frame is not None
        env_frame = np.array(frame[0], copy=True)
        actions_taken: list[int] = []
        for action in _DISPLAY_ACTIONS:
            _, _, term, trunc, _ = env.step(np.asarray([action], dtype=np.int32))
            actions_taken.append(action)
            if bool(term[0] or trunc[0]):
                break
            frame = env.render()
            assert frame is not None
            env_frame = np.array(frame[0], copy=True)
    finally:
        env.close()
        del env
        gc.collect()

    oracle = GfootballOracle(
        task_id,
        render=True,
        render_resolution_x=_CAPTURE_WIDTH,
        render_resolution_y=_CAPTURE_HEIGHT,
    )
    try:
        oracle.reset(
            engine_seed=_scalar(info["engine_seed"]),
            episode_number=_scalar(info["episode_number"]),
        )
        oracle_frame = oracle.render()
        for index, action in enumerate(actions_taken):
            oracle.step(action)
            if index + 1 == len(actions_taken):
                oracle_frame = oracle.render()
    finally:
        del oracle
        gc.collect()

    np.testing.assert_array_equal(env_frame, oracle_frame)
    return env_frame, oracle_frame


def _resize_frame(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(frame, mode="RGB").resize(
        (_FRAME_WIDTH, _FRAME_HEIGHT), Image.Resampling.BILINEAR
    )


def _scenario_label(task_id: str) -> str:
    scenario = task_id.removeprefix("gfootball/").removesuffix("-v1")
    return scenario.replace("_", " ")


def _panel_title_lines(task_id: str, width: int) -> list[str]:
    label = _scenario_label(task_id)
    lines = textwrap.wrap(label, width=width)
    return lines or [label]


def _draw_panel(
    canvas: Image.Image,
    task_id: str,
    env_frame: np.ndarray,
    official_frame: np.ndarray,
    origin: tuple[int, int],
    *,
    title_font: ImageFont.ImageFont,
    caption_font: ImageFont.ImageFont,
) -> None:
    x0, y0 = origin
    panel_width = 2 * _FRAME_WIDTH + _IMAGE_GAP + 2 * _PANEL_PADDING
    lines = _panel_title_lines(task_id, width=21)
    title_height = len(lines) * 16
    caption_height = 14
    panel_height = (
        2 * _PANEL_PADDING + title_height + caption_height + 8 + _FRAME_HEIGHT
    )
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        [x0, y0, x0 + panel_width, y0 + panel_height],
        radius=6,
        fill=_PANEL_BACKGROUND,
        outline=_PANEL_BORDER,
        width=1,
    )
    text_y = y0 + _PANEL_PADDING
    for line in lines:
        line_box = draw.textbbox((0, 0), line, font=title_font)
        line_width = line_box[2] - line_box[0]
        draw.text(
            (x0 + (panel_width - line_width) / 2, text_y),
            line,
            fill=_TEXT,
            font=title_font,
        )
        text_y += 16
    left_center = x0 + _PANEL_PADDING + _FRAME_WIDTH / 2
    right_center = left_center + _FRAME_WIDTH + _IMAGE_GAP
    for center_x, label in (
        (left_center, "EnvPool"),
        (right_center, "Official"),
    ):
        label_box = draw.textbbox((0, 0), label, font=caption_font)
        label_width = label_box[2] - label_box[0]
        draw.text(
            (center_x - label_width / 2, text_y),
            label,
            fill=_SUBTEXT,
            font=caption_font,
        )
    image_y = text_y + caption_height + 8
    canvas.paste(_resize_frame(env_frame), (x0 + _PANEL_PADDING, image_y))
    canvas.paste(
        _resize_frame(official_frame),
        (x0 + _PANEL_PADDING + _FRAME_WIDTH + _IMAGE_GAP, image_y),
    )


def build_image() -> Image.Image:
    """Build the combined render-compare image for all registered tasks."""
    register_gfootball_envs()
    panels: list[tuple[str, np.ndarray, np.ndarray]] = []
    for task_id in ALL_TASK_IDS:
        env_frame, official_frame = _capture_frames(task_id)
        panels.append((task_id, env_frame, official_frame))

    panel_width = 2 * _FRAME_WIDTH + _IMAGE_GAP + 2 * _PANEL_PADDING
    title_font = _load_font(14)
    caption_font = _load_font(12)
    rows = math.ceil(len(panels) / _GRID_COLUMNS)
    panel_height = 2 * _PANEL_PADDING + 32 + 14 + 8 + _FRAME_HEIGHT
    canvas_width = (
        2 * _OUTER_MARGIN
        + _GRID_COLUMNS * panel_width
        + (_GRID_COLUMNS - 1) * _PANEL_GAP_X
    )
    canvas_height = (
        2 * _OUTER_MARGIN
        + _HEADER_HEIGHT
        + rows * panel_height
        + (rows - 1) * _PANEL_GAP_Y
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), _BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    header_font = _load_font(16)
    header = (
        "EnvPool render on the left, independent official-engine oracle on "
        "the right"
    )
    draw.text(
        (_OUTER_MARGIN, _OUTER_MARGIN),
        header,
        fill=_TEXT,
        font=header_font,
    )
    y0 = _OUTER_MARGIN + _HEADER_HEIGHT
    for index, (task_id, env_frame, official_frame) in enumerate(panels):
        row, column = divmod(index, _GRID_COLUMNS)
        x = _OUTER_MARGIN + column * (panel_width + _PANEL_GAP_X)
        y = y0 + row * (panel_height + _PANEL_GAP_Y)
        _draw_panel(
            canvas,
            task_id,
            env_frame,
            official_frame,
            (x, y),
            title_font=title_font,
            caption_font=caption_font,
        )
    return canvas


def main() -> None:
    """Generate the render sample image on disk."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output
    build_working_directory = os.environ.get("BUILD_WORKING_DIRECTORY")
    if not output.is_absolute() and build_working_directory:
        output = Path(build_working_directory) / output
    image = build_image()
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)


if __name__ == "__main__":
    main()
