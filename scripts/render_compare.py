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
"""Generate render comparison and sample images for docs."""

from __future__ import annotations

import argparse
import math
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envpool.python.glfw_context import preload_windows_gl_dlls

preload_windows_gl_dlls(strict=True)


def _configure_linux_mujoco_gl() -> None:
    """Pick an offscreen MuJoCo GL backend on Linux before importing mujoco."""
    if platform.system() != "Linux" or os.environ.get("MUJOCO_GL"):
        return

    for backend in ("egl", "osmesa"):
        probe_env = os.environ.copy()
        probe_env["MUJOCO_GL"] = backend
        code = "import mujoco; mujoco.GLContext(8, 8).free()"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=probe_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode == 0:
            os.environ["MUJOCO_GL"] = backend
            return


_configure_linux_mujoco_gl()


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
    max_ignored_abs_diff: int
    require_bitwise: bool
    flip_vertical: bool


RenderPairFn = Callable[
    [str, RenderCompareConfig], tuple[np.ndarray, np.ndarray | None]
]


@dataclass(frozen=True)
class RenderItem:
    """One family-specific render key and its display label."""

    key: str
    label: str


@dataclass(frozen=True)
class RenderFamily:
    """Family-specific render integration."""

    items: tuple[RenderItem, ...]
    default_output: Path
    render_pair: RenderPairFn
    left_title: str = "EnvPool"
    right_title: str | None = "Official"
    default_flip_vertical: bool = False
    compare_frames: bool = True


_PAIR_GAP = 4
_CELL_GAP = 12
_HEADER_HEIGHT = 28
_MARGIN = 16


def _make_craftax_family() -> RenderFamily:
    """Build the Craftax comparison adapter using actual gameplay frames."""
    import envpool.craftax.registration  # noqa: F401
    from envpool.craftax.oracle import (
        encode,
        jax,
        make_oracle,
        native,
        renderer,
    )
    from envpool.registration import make_gymnasium

    scenes = {
        "Classic": (True, 0, 96),
        "Craftax": (False, 0, 96),
        "Cave": (False, 2, 48),
    }

    def render_pair(
        scene: str, cfg: RenderCompareConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        classic, level, length = scenes[scene]
        task = (
            "Craftax-"
            + ("Classic-" if classic else "")
            + "Symbolic-AutoReset-v1"
        )
        oracle, params = make_oracle(task)
        key = jax.random.PRNGKey(cfg.seed)
        _, state = oracle.reset(key, params)
        kwargs = {}
        if level:
            state = state.replace(
                player_level=jax.numpy.asarray(level, dtype=jax.numpy.int32),
                player_position=state.up_ladders[level],
            )
            layout = native.Game(native.Params(classic)).get_state()
            kwargs["initial_state"] = encode(state, layout).tolist()
        pool = make_gymnasium(
            task, num_envs=1, seed=cfg.seed, render_mode="rgb_array", **kwargs
        )
        step = jax.jit(oracle.step)
        rng = np.random.default_rng(40)
        try:
            pool.reset()
            for _ in range(length):
                key, draw = jax.random.split(key)
                action = int(rng.integers(17 if classic else 43))
                _, state, _, _, _ = step(draw, state, action, params)
                pool.step(np.array([action], np.int32))
            actual = np.asarray(pool.render())[0]
            expected = np.asarray(renderer(classic)(state)).astype(np.uint8)
            np.testing.assert_array_equal(actual, expected)
            return actual, expected
        finally:
            pool.close()

    return RenderFamily(
        items=tuple(
            RenderItem(key=scene, label=f"{scene} / {length} actions")
            for scene, (_, _, length) in scenes.items()
        ),
        default_output=Path(
            "docs/_static/render_samples/craftax_official_compare.png"
        ),
        render_pair=render_pair,
        right_title="Official Craftax 1.6.1",
    )


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


def _make_jumanji_family() -> RenderFamily:
    """Use the same reset synchronization as independent alignment tests."""
    import envpool.jumanji.registration as jumanji_registration
    from envpool.jumanji import oracle
    from envpool.registration import make_gymnasium

    def render_pair(
        task_id: str, cfg: RenderCompareConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        official_env = oracle._make_official_env(task_id)
        env = None
        try:
            state, timestep = official_env.reset(
                oracle.jax.random.PRNGKey(cfg.seed)
            )
            kwargs = oracle._envpool_kwargs_from_official_state(
                task_id, state, timestep.observation
            )
            if task_id == "Sokoban-v0":
                kwargs.update(
                    base_path="/tmp/envpool-missing-boxoban",
                    sokoban_level_index=0,
                )
            env = make_gymnasium(
                task_id,
                num_envs=1,
                seed=cfg.seed,
                render_mode="rgb_array",
                render_width=cfg.source_width,
                render_height=cfg.source_height,
                **kwargs,
            )
            env.reset()
            frames = env.render()
            assert frames is not None
            return frames[0], oracle._official_render(
                official_env, state, cfg.source_width, cfg.source_height
            )
        finally:
            if env is not None:
                env.close()
            official_env.close()

    return RenderFamily(
        items=tuple(
            RenderItem(key=task_id, label=task_id)
            for task_id in jumanji_registration.jumanji_env_ids
        ),
        default_output=Path(
            "docs/_static/render_samples/jumanji_official_compare.png"
        ),
        render_pair=render_pair,
        left_title="EnvPool",
        right_title="Official",
        # Native task graphics intentionally use a different layout and style.
        compare_frames=False,
    )


def _make_mujoco_playground_family() -> RenderFamily:
    """Build the MuJoCo Playground official render comparison adapter."""
    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)

    import mujoco

    import envpool.mujoco.playground.registration as playground_registration
    from envpool.mujoco.playground.oracle import FrameRenderer, make_oracle
    from envpool.registration import make_gymnasium

    def make_render_data(model: Any, info: dict[str, Any]) -> Any:
        data = mujoco.MjData(model)
        data.qpos[:] = info["qpos"][0, : model.nq]
        data.qvel[:] = info["qvel"][0, : model.nv]
        if "ctrl" in info:
            data.ctrl[:] = info["ctrl"][0, : model.nu]
        if "mocap_pos" in info:
            data.mocap_pos[:] = info["mocap_pos"][
                0, : model.nmocap * 3
            ].reshape(model.nmocap, 3)
        if "mocap_quat" in info:
            data.mocap_quat[:] = info["mocap_quat"][
                0, : model.nmocap * 4
            ].reshape(model.nmocap, 4)
        if "qacc_warmstart" in info:
            data.qacc_warmstart[:] = info["qacc_warmstart"][0, : model.nv]
        mujoco.mj_forward(model, data)
        if "qacc" in info:
            data.qacc[:] = info["qacc"][0, : model.nv]
        if "qacc_warmstart" in info:
            data.qacc_warmstart[:] = info["qacc_warmstart"][0, : model.nv]
        if "sensordata" in info:
            data.sensordata[:] = info["sensordata"][0, : data.sensordata.size]
        return data

    def zero_command_kwargs(
        *,
        noise_key: str | None = "noise_level",
        noise_value: float = 0.0,
        disable_key: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "lin_vel_x_min": 0.0,
            "lin_vel_x_max": 0.0,
            "lin_vel_y_min": 0.0,
            "lin_vel_y_max": 0.0,
            "ang_vel_yaw_min": 0.0,
            "ang_vel_yaw_max": 0.0,
        }
        if noise_key is not None:
            kwargs[noise_key] = noise_value
        if disable_key is not None:
            kwargs[disable_key] = 0.0
        return kwargs

    def envpool_kwargs(task_id: str) -> dict[str, Any]:
        if task_id.startswith("Apollo"):
            return {
                "noise_level": 0.0,
                "push_enable": 0.0,
                "command_min0": 0.0,
                "command_min1": 0.0,
                "command_min2": 0.0,
                "command_max0": 0.0,
                "command_max1": 0.0,
                "command_max2": 0.0,
                "command_zero_prob0": 1.0,
                "command_zero_prob1": 1.0,
                "command_zero_prob2": 1.0,
            }
        if task_id.startswith("Barkour"):
            return {
                **zero_command_kwargs(noise_key="obs_noise", noise_value=-1.0),
                "kick_wait_steps_min": 100000,
                "kick_wait_steps_max": 100001,
            }
        if task_id.startswith("Op3"):
            return zero_command_kwargs(noise_key="obs_noise", noise_value=-1.0)
        if task_id.startswith(("BerkeleyHumanoid", "G1", "T1")):
            return zero_command_kwargs(disable_key="push_enable")
        if task_id.startswith("H1"):
            return zero_command_kwargs(noise_key="obs_noise_level")
        if task_id == "SpotFlatTerrainJoystick-v1":
            return zero_command_kwargs(disable_key="pert_enable")
        if task_id == "SpotGetup-v1":
            return {"noise_level": 0.0}
        if task_id == "SpotJoystickGaitTracking-v1":
            return zero_command_kwargs()
        if task_id == "PandaPickCubeCartesian-v1":
            return {"guide_sample_prob": 0.0}
        if task_id == "PandaRobotiqPushCube-v1":
            return {
                "action_min_delay": 0,
                "action_max_delay": 1,
                "obs_min_delay": 0,
                "obs_max_delay": 1,
                "noise_obj_pos": 0.0,
                "noise_obj_angle": 0.0,
                "noise_robot_qpos": 0.0,
                "noise_robot_qvel": 0.0,
                "noise_eef_pos": 0.0,
                "noise_eef_angle": 0.0,
            }
        if task_id in (
            "LeapCubeRotateZAxis-v1",
            "LeapCubeReorient-v1",
            "AeroCubeRotateZAxis-v1",
        ):
            return {"noise_level": 0.0, "pert_enable": 0.0}
        return {}

    def render_pair(
        task_id: str,
        cfg: RenderCompareConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        oracle = make_oracle(task_id)
        env = make_gymnasium(
            task_id,
            num_envs=1,
            seed=cfg.seed,
            render_mode="rgb_array",
            render_width=cfg.source_width,
            render_height=cfg.source_height,
            **envpool_kwargs(task_id),
        )
        try:
            _, info = env.reset()
            render_model = oracle.mj_model
            render_data = make_render_data(render_model, info)
            renderer = FrameRenderer(
                oracle,
                task_id,
                width=cfg.source_width,
                height=cfg.source_height,
            )
            try:
                official_frame = renderer.render(render_data)
            finally:
                renderer.close()
            envpool_frame = env.render(env_ids=[0])
            if envpool_frame is None:
                raise RuntimeError(f"{task_id} returned no EnvPool frame")
            return np.asarray(envpool_frame[0]), official_frame
        finally:
            env.close()

    return RenderFamily(
        items=tuple(
            RenderItem(
                key=f"{task_name}-v1",
                label=task_name
                .replace("BerkeleyHumanoid", "Berkeley ")
                .replace("G1", "G1 ")
                .replace("Go1", "Go1 ")
                .replace("Op3", "OP3"),
            )
            for task_name in playground_registration.PLAYGROUND_ENVS
        ),
        default_output=Path(
            "docs/_static/render_samples/mujoco_playground_official_compare.png"
        ),
        render_pair=render_pair,
    )


_FAMILY_BUILDERS: dict[str, Callable[[], RenderFamily]] = {
    "craftax": _make_craftax_family,
    "jumanji": _make_jumanji_family,
    "metaworld": _make_metaworld_family,
    "mujoco_playground": _make_mujoco_playground_family,
}


def _make_display_image(
    frame: np.ndarray, cfg: RenderCompareConfig
) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    if image.mode != "RGB":
        image = image.convert("RGB")
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
    max_ignored_abs_diff: int,
) -> str:
    return (
        f"{label} render mismatch: "
        f"mean_abs_diff={float(diff.mean()):.6f}, "
        f"max_abs_diff={int(diff.max())}, "
        f"mismatch_ratio={mismatch_ratio:.6%}, "
        f"ignored_abs_diff={max_ignored_abs_diff}"
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
            _render_mismatch_message(label, diff, mismatch_ratio, 0)
        )

    mean_abs_diff = float(diff.mean())
    significant = diff > cfg.max_ignored_abs_diff
    mismatch_ratio = float(np.count_nonzero(significant)) / float(diff.size)
    if (
        mean_abs_diff > cfg.max_mean_abs_diff
        or mismatch_ratio > cfg.max_mismatch_ratio
    ):
        raise RuntimeError(
            _render_mismatch_message(
                label, diff, mismatch_ratio, cfg.max_ignored_abs_diff
            )
        )


def _draw_panel(
    canvas: Image.Image,
    family: RenderFamily,
    label: str,
    envpool_image: Image.Image,
    official_image: Image.Image | None,
    index: int,
    cfg: RenderCompareConfig,
    font: Any,
) -> None:
    draw = ImageDraw.Draw(canvas)
    has_official = official_image is not None
    cell_width = cfg.tile_width * (2 if has_official else 1)
    if has_official:
        cell_width += _PAIR_GAP
    cell_height = _HEADER_HEIGHT + cfg.tile_height
    col = index % cfg.columns
    row = index // cfg.columns
    left = _MARGIN + col * (cell_width + _CELL_GAP)
    top = _MARGIN + row * (cell_height + _CELL_GAP)
    frame_top = top + _HEADER_HEIGHT

    draw.text((left, top), label, fill=(30, 30, 30), font=font)
    draw.text((left, top + 14), family.left_title, fill=(80, 80, 80), font=font)
    canvas.paste(envpool_image, (left, frame_top))
    draw.rectangle(
        [
            left,
            frame_top,
            left + cfg.tile_width - 1,
            frame_top + cfg.tile_height - 1,
        ],
        outline=(205, 205, 205),
    )
    if has_official:
        official_left = left + cfg.tile_width + _PAIR_GAP
        assert family.right_title is not None
        draw.text(
            (official_left, top + 14),
            family.right_title,
            fill=(80, 80, 80),
            font=font,
        )
        canvas.paste(official_image, (official_left, frame_top))
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
    """Generate and write one docs render image."""
    rows = math.ceil(len(family.items) / cfg.columns)
    has_official = family.right_title is not None
    cell_width = cfg.tile_width * (2 if has_official else 1)
    if has_official:
        cell_width += _PAIR_GAP
    cell_height = _HEADER_HEIGHT + cfg.tile_height
    width = _MARGIN * 2 + cfg.columns * cell_width
    width += (cfg.columns - 1) * _CELL_GAP
    height = _MARGIN * 2 + rows * cell_height + (rows - 1) * _CELL_GAP
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    font = ImageFont.load_default()

    for index, item in enumerate(family.items):
        envpool_frame, official_frame = family.render_pair(item.key, cfg)
        if official_frame is not None and family.compare_frames:
            _assert_frames_match(item.label, envpool_frame, official_frame, cfg)
        envpool_image = _make_display_image(envpool_frame, cfg)
        official_image = (
            None
            if official_frame is None
            else _make_display_image(official_frame, cfg)
        )
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
    parser.add_argument(
        "--max-ignored-abs-diff",
        type=int,
        default=0,
        help=(
            "RGB channel deltas up to this value are ignored when computing "
            "mismatch ratio. Mean absolute diff still uses raw deltas."
        ),
    )
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
        max_ignored_abs_diff=args.max_ignored_abs_diff,
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
