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
"""Generate EnvPool-vs-oracle render comparison images for docs."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envpool.python.glfw_context import preload_windows_gl_dlls

preload_windows_gl_dlls(strict=True)


@dataclass(frozen=True)
class RenderCompareConfig:
    """Shared image generation and comparison settings."""

    family: str
    tile_width: int
    tile_height: int
    source_width: int
    source_height: int
    columns: int
    seed: int
    camera_id: int
    max_mean_abs_diff: float
    max_mismatch_ratio: float
    require_bitwise: bool
    flip_vertical: bool


RenderPairFn = Callable[
    [str, RenderCompareConfig], tuple[np.ndarray, np.ndarray]
]


@dataclass(frozen=True)
class RenderItem:
    """One family-specific oracle key and its display label."""

    key: str
    label: str
    max_mean_abs_diff: float | None = None
    max_mismatch_ratio: float | None = None
    require_match: bool = True


@dataclass(frozen=True)
class RenderFamily:
    """Family-specific render oracle integration."""

    items: tuple[RenderItem, ...]
    default_output: Path
    render_pair: RenderPairFn
    left_title: str = "EnvPool"
    right_title: str = "Official"
    default_flip_vertical: bool = False


_PAIR_GAP = 4
_CELL_GAP = 12
_HEADER_HEIGHT = 28
_MARGIN = 16


def _make_metaworld_family() -> RenderFamily:
    """Build the MetaWorld render comparison adapter."""
    import mujoco
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS

    import envpool.mujoco.metaworld.registration as metaworld_registration
    from envpool.registration import make_gymnasium

    def make_oracle(task_name: str, cfg: RenderCompareConfig) -> Any:
        env = ALL_V3_ENVIRONMENTS[task_name](
            render_mode="rgb_array",
            camera_id=cfg.camera_id,
        )
        env._set_task_called = True
        env._partially_observable = True
        env.mujoco_renderer.width = cfg.source_width
        env.mujoco_renderer.height = cfg.source_height
        return env

    def sync_reset_state(oracle: Any, info: dict[str, Any]) -> None:
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
                "`bazel run --config=debug //scripts:render_compare -- "
                "--family=metaworld` so EnvPool exposes reset-sync info. "
                f"Missing keys: {missing}"
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
        oracle.init_tcp = np.asarray(
            info["init_tcp0"][0], dtype=np.float64
        ).copy()
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

    def render_pair(
        task_name: str,
        cfg: RenderCompareConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        task_id = metaworld_registration.metaworld_task_id(task_name)
        env = make_gymnasium(
            task_id,
            num_envs=1,
            seed=cfg.seed,
            render_mode="rgb_array",
            render_width=cfg.source_width,
            render_height=cfg.source_height,
            render_camera_id=cfg.camera_id,
        )
        oracle = make_oracle(task_name, cfg)
        try:
            _, info = env.reset()
            sync_reset_state(oracle, info)
            envpool_frame = env.render()
            if envpool_frame is None:
                raise RuntimeError(f"{task_id} returned no EnvPool frame")
            return envpool_frame[0], oracle.render()
        finally:
            env.close()
            oracle.close()

    return RenderFamily(
        items=tuple(
            RenderItem(
                key=task_name,
                label=metaworld_registration.metaworld_public_task_name(
                    task_name
                ),
            )
            for task_name in metaworld_registration.metaworld_v3_envs
        ),
        default_output=Path(
            "docs/_static/render_samples/metaworld_official_compare.png"
        ),
        render_pair=render_pair,
        default_flip_vertical=True,
    )


def _make_myosuite_family() -> RenderFamily:
    """Build the MyoSuite render comparison adapter."""
    from envpool.mujoco.myosuite.render_utils import (
        MYOSUITE_RENDER_COMPARE_CASES,
        MYOSUITE_RENDER_COMPARE_STEPS,
        MYOSUITE_RENDER_RETRY_SEEDS,
        capture_render_sequence,
        official_render_thresholds,
    )

    cases = tuple(MYOSUITE_RENDER_COMPARE_CASES)
    sequence_cache: dict[str, Any] = {}
    case_thresholds = {
        case.task_id: official_render_thresholds(case.task_id) for case in cases
    }

    def _split_case(key: str) -> tuple[str, int]:
        task_id, step_index = key.split(":", maxsplit=1)
        return task_id, int(step_index)

    def render_pair(
        key: str,
        cfg: RenderCompareConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        task_id, step_index = _split_case(key)
        sequence = sequence_cache.get(task_id)
        if sequence is None:
            sequence = capture_render_sequence(
                task_id,
                steps=MYOSUITE_RENDER_COMPARE_STEPS,
                seed=cfg.seed,
                render_width=cfg.source_width,
                render_height=cfg.source_height,
                camera_id=cfg.camera_id,
                retry_seeds=MYOSUITE_RENDER_RETRY_SEEDS,
            )
            sequence_cache[task_id] = sequence
        return (
            sequence.envpool_frames[step_index],
            sequence.official_frames[step_index],
        )

    return RenderFamily(
        items=tuple(
            RenderItem(
                key=f"{case.task_id}:{step_index}",
                label=f"{case.label} step {step_index + 1}",
                max_mean_abs_diff=(
                    case_thresholds[case.task_id][0]
                    if case_thresholds[case.task_id] is not None
                    else None
                ),
                max_mismatch_ratio=(
                    case_thresholds[case.task_id][1]
                    if case_thresholds[case.task_id] is not None
                    else None
                ),
                require_match=True,
            )
            for case in cases
            for step_index in range(MYOSUITE_RENDER_COMPARE_STEPS)
        ),
        default_output=Path(
            "docs/_static/render_samples/myosuite_official_compare.png"
        ),
        render_pair=render_pair,
    )


_FAMILY_BUILDERS: dict[str, Callable[[], RenderFamily]] = {
    "metaworld": _make_metaworld_family,
    "myosuite": _make_myosuite_family,
}


def _make_display_image(
    frame: np.ndarray, cfg: RenderCompareConfig
) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    if cfg.flip_vertical:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    size = (cfg.tile_width, cfg.tile_height)
    if image.size == size:
        return image
    return image.resize(size, Image.Resampling.LANCZOS)


def _render_mismatch_message(
    label: str,
    diff: np.ndarray,
    mismatch_ratio: float,
) -> str:
    return (
        f"{label} render mismatch: "
        f"mean_abs_diff={float(diff.mean()):.6f}, "
        f"max_abs_diff={int(diff.max())}, "
        f"mismatch_ratio={mismatch_ratio:.6%}"
    )


def _assert_frames_match(
    label: str,
    envpool_frame: np.ndarray,
    official_frame: np.ndarray,
    cfg: RenderCompareConfig,
) -> None:
    if envpool_frame.shape != official_frame.shape:
        raise RuntimeError(
            f"{label} render shape mismatch: "
            f"{envpool_frame.shape} != {official_frame.shape}"
        )
    if np.array_equal(envpool_frame, official_frame):
        return

    diff = np.abs(
        envpool_frame.astype(np.int16) - official_frame.astype(np.int16)
    )
    mismatch_ratio = float(np.count_nonzero(diff)) / float(diff.size)
    if cfg.require_bitwise:
        raise RuntimeError(
            _render_mismatch_message(label, diff, mismatch_ratio)
        )

    mean_abs_diff = float(diff.mean())
    if (
        mean_abs_diff > cfg.max_mean_abs_diff
        or mismatch_ratio > cfg.max_mismatch_ratio
    ):
        raise RuntimeError(
            _render_mismatch_message(label, diff, mismatch_ratio)
        )


def _draw_panel(
    canvas: Image.Image,
    family: RenderFamily,
    label: str,
    envpool_image: Image.Image,
    official_image: Image.Image,
    index: int,
    cfg: RenderCompareConfig,
    font: Any,
) -> None:
    draw = ImageDraw.Draw(canvas)
    cell_width = cfg.tile_width * 2 + _PAIR_GAP
    cell_height = _HEADER_HEIGHT + cfg.tile_height
    col = index % cfg.columns
    row = index // cfg.columns
    left = _MARGIN + col * (cell_width + _CELL_GAP)
    top = _MARGIN + row * (cell_height + _CELL_GAP)
    frame_top = top + _HEADER_HEIGHT
    official_left = left + cfg.tile_width + _PAIR_GAP

    draw.text((left, top), label, fill=(30, 30, 30), font=font)
    draw.text((left, top + 14), family.left_title, fill=(80, 80, 80), font=font)
    draw.text(
        (official_left, top + 14),
        family.right_title,
        fill=(80, 80, 80),
        font=font,
    )
    canvas.paste(envpool_image, (left, frame_top))
    canvas.paste(official_image, (official_left, frame_top))
    draw.rectangle(
        [
            left,
            frame_top,
            left + cfg.tile_width - 1,
            frame_top + cfg.tile_height - 1,
        ],
        outline=(205, 205, 205),
    )
    draw.rectangle(
        [
            official_left,
            frame_top,
            official_left + cfg.tile_width - 1,
            frame_top + cfg.tile_height - 1,
        ],
        outline=(205, 205, 205),
    )


def generate(
    output: Path, family: RenderFamily, cfg: RenderCompareConfig
) -> None:
    """Generate and write one docs render comparison image."""
    rows = math.ceil(len(family.items) / cfg.columns)
    cell_width = cfg.tile_width * 2 + _PAIR_GAP
    cell_height = _HEADER_HEIGHT + cfg.tile_height
    width = _MARGIN * 2 + cfg.columns * cell_width
    width += (cfg.columns - 1) * _CELL_GAP
    height = _MARGIN * 2 + rows * cell_height + (rows - 1) * _CELL_GAP
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    font = ImageFont.load_default()

    for index, item in enumerate(family.items):
        envpool_frame, official_frame = family.render_pair(item.key, cfg)
        if item.require_match:
            item_cfg = RenderCompareConfig(
                family=cfg.family,
                tile_width=cfg.tile_width,
                tile_height=cfg.tile_height,
                source_width=cfg.source_width,
                source_height=cfg.source_height,
                columns=cfg.columns,
                seed=cfg.seed,
                camera_id=cfg.camera_id,
                max_mean_abs_diff=(
                    cfg.max_mean_abs_diff
                    if item.max_mean_abs_diff is None
                    else item.max_mean_abs_diff
                ),
                max_mismatch_ratio=(
                    cfg.max_mismatch_ratio
                    if item.max_mismatch_ratio is None
                    else item.max_mismatch_ratio
                ),
                require_bitwise=cfg.require_bitwise,
                flip_vertical=cfg.flip_vertical,
            )
            _assert_frames_match(
                item.label,
                envpool_frame,
                official_frame,
                item_cfg,
            )
        envpool_image = _make_display_image(envpool_frame, cfg)
        official_image = _make_display_image(official_frame, cfg)
        _draw_panel(
            canvas,
            family,
            item.label,
            envpool_image,
            official_image,
            index,
            cfg,
            font,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        choices=sorted(_FAMILY_BUILDERS),
        default="metaworld",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tile-width", type=int, default=96)
    parser.add_argument("--tile-height", type=int, default=72)
    parser.add_argument(
        "--source-width",
        type=int,
        default=480,
        help="Source render width before docs thumbnail downsampling.",
    )
    parser.add_argument(
        "--source-height",
        type=int,
        default=480,
        help="Source render height before docs thumbnail downsampling.",
    )
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--camera-id",
        type=int,
        default=1,
        help="Renderer camera id for families that support camera selection.",
    )
    parser.add_argument("--max-mean-abs-diff", type=float, default=0.25)
    parser.add_argument("--max-mismatch-ratio", type=float, default=0.005)
    parser.add_argument("--require-bitwise", action="store_true")
    parser.add_argument(
        "--flip-vertical",
        action="store_true",
        default=None,
        help="Flip both EnvPool and oracle frames vertically in the output.",
    )
    parser.add_argument(
        "--no-flip-vertical",
        action="store_false",
        dest="flip_vertical",
        help="Do not vertically flip output frames.",
    )
    return parser.parse_args()


def main() -> None:
    """Parse command-line arguments and generate the comparison image."""
    args = _parse_args()
    family = _FAMILY_BUILDERS[args.family]()
    cfg = RenderCompareConfig(
        family=args.family,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        source_width=args.source_width,
        source_height=args.source_height,
        columns=args.columns,
        seed=args.seed,
        camera_id=args.camera_id,
        max_mean_abs_diff=args.max_mean_abs_diff,
        max_mismatch_ratio=args.max_mismatch_ratio,
        require_bitwise=args.require_bitwise,
        flip_vertical=(
            family.default_flip_vertical
            if args.flip_vertical is None
            else args.flip_vertical
        ),
    )

    output = args.output or family.default_output
    if not output.is_absolute():
        output = Path(
            os.environ.get("BUILD_WORKSPACE_DIRECTORY", ".")
        ).joinpath(output)
    generate(output, family, cfg)


if __name__ == "__main__":
    main()
