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
"""Generate MyoSuite EnvPool-vs-official render samples for docs."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _bootstrap_envpool_namespace() -> None:
    """Load envpool modules without importing envpool.entry."""
    if "envpool" in sys.modules:
        return
    roots = []
    for path in sys.path:
        envpool_root = Path(path) / "envpool"
        if (
            envpool_root / "registration.py"
        ).is_file() or envpool_root.is_dir():
            roots.append(str(envpool_root))
    if not roots:
        raise RuntimeError("could not locate envpool package on PYTHONPATH")
    module = types.ModuleType("envpool")
    module.__file__ = str(Path(roots[0]) / "__init__.py")
    module.__path__ = roots  # type: ignore[attr-defined]
    sys.modules["envpool"] = module


def _make_gymnasium(task_id: str, **kwargs: object):
    _bootstrap_envpool_namespace()
    if not getattr(_make_gymnasium, "_registered", False):
        importlib.import_module("envpool.mujoco.myosuite.registration")
        _make_gymnasium._registered = True  # type: ignore[attr-defined]
    from envpool.registration import make_gymnasium

    return make_gymnasium(task_id, **kwargs)


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    draw.rectangle(
        (xy[0] - 3, xy[1] - 2, xy[0] + 156, xy[1] + 14),
        fill=(255, 255, 255),
    )
    draw.text(xy, text, fill=(0, 0, 0))


def _registered_tasks() -> tuple[dict[str, object], ...]:
    _bootstrap_envpool_namespace()
    from envpool.mujoco.myosuite.tasks import MYOSUITE_TASKS

    return tuple(MYOSUITE_TASKS)


def _group_name(task_id: str) -> str:
    if task_id.startswith("MyoHand"):
        return "myodm"
    if "Challenge" in task_id:
        return "myochallenge"
    return "myobase"


def _envpool_frames(
    task_id: str,
    width: int,
    height: int,
    seed: int,
    actions: Sequence[Sequence[float]],
) -> tuple[list[np.ndarray], list[Mapping[str, object]]]:
    env = _make_gymnasium(
        task_id,
        num_envs=1,
        seed=seed,
        render_mode="rgb_array",
        render_width=width,
        render_height=height,
    )
    try:
        _, info = env.reset()
        frames = [env.render()[0]]
        infos: list[Mapping[str, object]] = [info]
        for action in actions:
            *_, info = env.step(np.asarray(action, dtype=np.float32)[None, :])
            frames.append(env.render()[0])
            infos.append(info)
        return frames, infos
    finally:
        env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle_trace", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--all_tasks", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--require_bitwise", action="store_true")
    parser.add_argument("--task_id", default="myoFingerReachFixed-v0")
    parser.add_argument("--seed", default=3, type=int)
    parser.add_argument("--width", default=160, type=int)
    parser.add_argument("--height", default=120, type=int)
    parser.add_argument("--camera_id", default=-1, type=int)
    return parser.parse_args()


def _official_oracle() -> object:
    _bootstrap_envpool_namespace()
    oracle = importlib.import_module(
        "envpool.mujoco.myosuite.myosuite_oracle_probe"
    )
    oracle._configure_linux_mujoco_renderer(True)  # type: ignore[attr-defined]
    return oracle


def _official_gym() -> object:
    if not hasattr(_official_gym, "_gym"):
        oracle = _official_oracle()
        _, _, gym = oracle._import_official()  # type: ignore[attr-defined]
        _official_gym._gym = gym  # type: ignore[attr-defined]
    return _official_gym._gym  # type: ignore[attr-defined]


def _task_frames_from_trace(
    task_id: str,
    task_trace: Mapping[str, object],
    width: int,
    height: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[Mapping[str, object]]]:
    oracle_frames = [
        np.asarray(frame, dtype=np.uint8)
        for frame in task_trace["frames"]  # type: ignore[index]
    ]
    envpool_frames, envpool_infos = _envpool_frames(
        task_id,
        width,
        height,
        seed,
        task_trace["actions"],  # type: ignore[arg-type,index]
    )
    if len(envpool_frames) != 4 or len(oracle_frames) != 4:
        raise ValueError("expected reset plus three step frames")
    return envpool_frames, oracle_frames, envpool_infos


def _task_frames_from_official(
    task_id: str,
    width: int,
    height: int,
    seed: int,
    camera_id: int,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[Mapping[str, object]],
    Mapping[str, object],
]:
    oracle = _official_oracle()
    gym = _official_gym()
    rng = np.random.default_rng(seed + 17)
    env = gym.make(task_id)  # type: ignore[attr-defined]
    try:
        reset = env.reset(seed=seed)
        obs = reset[0] if isinstance(reset, tuple) else reset
        unwrapped = env.unwrapped
        low = np.asarray(env.action_space.low, dtype=np.float32)
        high = np.asarray(env.action_space.high, dtype=np.float32)
        oracle_frames = [
            np.asarray(
                oracle._render_frame(  # type: ignore[attr-defined]
                    env, width, height, camera_id
                ),
                dtype=np.uint8,
            )
        ]
        task_trace: dict[str, object] = {
            "actions": [],
            "obs": [oracle._jsonable_array(obs)],  # type: ignore[attr-defined]
            "reset_state": oracle._state_report(unwrapped),  # type: ignore[attr-defined]
            "states": [],
        }
        for _ in range(3):
            action = rng.uniform(low, high).astype(np.float32)
            step = env.step(action)
            task_trace["actions"].append(  # type: ignore[union-attr]
                oracle._jsonable_array(action)  # type: ignore[attr-defined]
            )
            task_trace["obs"].append(  # type: ignore[union-attr]
                oracle._jsonable_array(step[0])  # type: ignore[attr-defined]
            )
            task_trace["states"].append(  # type: ignore[union-attr]
                oracle._state_report(unwrapped)  # type: ignore[attr-defined]
            )
            oracle_frames.append(
                np.asarray(
                    oracle._render_frame(  # type: ignore[attr-defined]
                        env, width, height, camera_id
                    ),
                    dtype=np.uint8,
                )
            )
        envpool_frames, envpool_infos = _envpool_frames(
            task_id,
            width,
            height,
            seed,
            task_trace["actions"],  # type: ignore[arg-type]
        )
        return envpool_frames, oracle_frames, envpool_infos, task_trace
    finally:
        env.close()


def _bitwise_stats(
    task_id: str,
    envpool_frames: Sequence[np.ndarray],
    oracle_frames: Sequence[np.ndarray],
    task_trace: Mapping[str, object],
    envpool_infos: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    diffs = [
        np.abs(a.astype(np.int16) - b.astype(np.int16))
        for a, b in zip(envpool_frames, oracle_frames, strict=True)
    ]
    stats: dict[str, object] = {
        "task_id": task_id,
        "frames": len(diffs),
        "max_abs_diff": max(int(diff.max()) for diff in diffs),
        "nonzero_channels": sum(int(np.count_nonzero(diff)) for diff in diffs),
    }
    states = [task_trace.get("reset_state", {})] + list(
        task_trace.get("states", [])  # type: ignore[arg-type]
    )
    state_diffs: dict[str, list[float]] = {}
    for info, state in zip(envpool_infos, states, strict=False):
        if not isinstance(state, Mapping) or "qpos" not in state:
            continue
        for key in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart"):
            if key not in info or key not in state:
                continue
            value = np.asarray(info[key], dtype=np.float64).ravel()
            oracle_value = np.asarray(state[key], dtype=np.float64).ravel()
            if oracle_value.size == 0:
                continue
            state_diffs.setdefault(key, []).append(
                float(np.max(np.abs(value[: oracle_value.size] - oracle_value)))
            )
    for key, diffs_for_key in state_diffs.items():
        stats[f"max_{key}_abs_diff"] = max(diffs_for_key)
    return stats


def _render_single(
    task_id: str,
    task_trace: Mapping[str, object],
    out: Path,
    width: int,
    height: int,
    seed: int,
) -> dict[str, object]:
    envpool_frames, oracle_frames, envpool_infos = _task_frames_from_trace(
        task_id, task_trace, width, height, seed
    )
    margin = 18
    label_h = 20
    gutter = 12
    row_gap = 10
    cell_w = width
    cell_h = height + label_h
    canvas_w = margin * 2 + cell_w * 2 + gutter
    canvas_h = margin * 2 + cell_h * 4 + row_gap * 3
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    labels = ["reset", "step 1", "step 2", "step 3"]
    for idx, label in enumerate(labels):
        y = margin + idx * (cell_h + row_gap)
        left_x = margin
        right_x = margin + cell_w + gutter
        _label(draw, (left_x, y), f"EnvPool {label}")
        _label(draw, (right_x, y), f"Official {label}")
        canvas.paste(
            Image.fromarray(envpool_frames[idx]), (left_x, y + label_h)
        )
        canvas.paste(
            Image.fromarray(oracle_frames[idx]), (right_x, y + label_h)
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return _bitwise_stats(
        task_id, envpool_frames, oracle_frames, task_trace, envpool_infos
    )


def _thumbnail(frame: np.ndarray, width: int, height: int) -> Image.Image:
    image = Image.fromarray(frame)
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _render_group(
    group: str,
    task_ids: Sequence[str],
    traces: Mapping[str, Mapping[str, object]],
    out_dir: Path,
    width: int,
    height: int,
    seed: int,
) -> list[dict[str, object]]:
    labels = ("reset", "step 1", "step 2", "step 3")
    margin = 12
    label_w = 238
    label_h = 18
    gutter = 4
    row_gap = 6
    header_h = 36
    thumb_w = width
    thumb_h = height
    row_h = thumb_h + label_h + row_gap
    canvas_w = margin * 2 + label_w + (thumb_w + gutter) * 8 - gutter
    canvas_h = margin * 2 + header_h + row_h * len(task_ids)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, margin),
        f"{group}: EnvPool left, Official right; reset + first 3 steps",
        fill=(0, 0, 0),
    )
    for idx, label in enumerate(labels):
        x = margin + label_w + idx * 2 * (thumb_w + gutter)
        draw.text((x, margin + label_h), f"{label} Env", fill=(0, 0, 0))
        draw.text(
            (x + thumb_w + gutter, margin + label_h),
            f"{label} Official",
            fill=(0, 0, 0),
        )

    stats: list[dict[str, object]] = []
    for row, task_id in enumerate(task_ids):
        y = margin + header_h + row * row_h
        draw.text((margin, y + label_h), task_id[:36], fill=(0, 0, 0))
        envpool_frames, oracle_frames, envpool_infos = _task_frames_from_trace(
            task_id, traces[task_id], width, height, seed
        )
        stats.append(
            _bitwise_stats(
                task_id,
                envpool_frames,
                oracle_frames,
                traces[task_id],
                envpool_infos,
            )
        )
        for idx in range(4):
            x = margin + label_w + idx * 2 * (thumb_w + gutter)
            canvas.paste(
                _thumbnail(envpool_frames[idx], thumb_w, thumb_h), (x, y)
            )
            canvas.paste(
                _thumbnail(oracle_frames[idx], thumb_w, thumb_h),
                (x + thumb_w + gutter, y),
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(out_dir / f"myosuite_{group}_official_compare.png")
    return stats


def _render_group_from_official(
    group: str,
    task_ids: Sequence[str],
    out_dir: Path,
    width: int,
    height: int,
    seed: int,
    camera_id: int,
) -> list[dict[str, object]]:
    labels = ("reset", "step 1", "step 2", "step 3")
    margin = 12
    label_w = 238
    label_h = 18
    gutter = 4
    row_gap = 6
    header_h = 36
    thumb_w = width
    thumb_h = height
    row_h = thumb_h + label_h + row_gap
    canvas_w = margin * 2 + label_w + (thumb_w + gutter) * 8 - gutter
    canvas_h = margin * 2 + header_h + row_h * len(task_ids)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, margin),
        f"{group}: EnvPool left, Official right; reset + first 3 steps",
        fill=(0, 0, 0),
    )
    for idx, label in enumerate(labels):
        x = margin + label_w + idx * 2 * (thumb_w + gutter)
        draw.text((x, margin + label_h), f"{label} Env", fill=(0, 0, 0))
        draw.text(
            (x + thumb_w + gutter, margin + label_h),
            f"{label} Official",
            fill=(0, 0, 0),
        )

    stats: list[dict[str, object]] = []
    for row, task_id in enumerate(task_ids):
        y = margin + header_h + row * row_h
        draw.text((margin, y + label_h), task_id[:36], fill=(0, 0, 0))
        (
            envpool_frames,
            oracle_frames,
            envpool_infos,
            task_trace,
        ) = _task_frames_from_official(task_id, width, height, seed, camera_id)
        stats.append(
            _bitwise_stats(
                task_id,
                envpool_frames,
                oracle_frames,
                task_trace,
                envpool_infos,
            )
        )
        for idx in range(4):
            x = margin + label_w + idx * 2 * (thumb_w + gutter)
            canvas.paste(
                _thumbnail(envpool_frames[idx], thumb_w, thumb_h), (x, y)
            )
            canvas.paste(
                _thumbnail(oracle_frames[idx], thumb_w, thumb_h),
                (x + thumb_w + gutter, y),
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(out_dir / f"myosuite_{group}_official_compare.png")
    return stats


def _render_single_from_official(
    task_id: str,
    out: Path,
    width: int,
    height: int,
    seed: int,
    camera_id: int,
) -> dict[str, object]:
    envpool_frames, oracle_frames, envpool_infos, task_trace = (
        _task_frames_from_official(task_id, width, height, seed, camera_id)
    )
    margin = 18
    label_h = 20
    gutter = 12
    row_gap = 10
    cell_w = width
    cell_h = height + label_h
    canvas_w = margin * 2 + cell_w * 2 + gutter
    canvas_h = margin * 2 + cell_h * 4 + row_gap * 3
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["reset", "step 1", "step 2", "step 3"]
    for idx, label in enumerate(labels):
        y = margin + idx * (cell_h + row_gap)
        left_x = margin
        right_x = margin + cell_w + gutter
        _label(draw, (left_x, y), f"EnvPool {label}")
        _label(draw, (right_x, y), f"Official {label}")
        canvas.paste(
            Image.fromarray(envpool_frames[idx]), (left_x, y + label_h)
        )
        canvas.paste(
            Image.fromarray(oracle_frames[idx]), (right_x, y + label_h)
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return _bitwise_stats(
        task_id, envpool_frames, oracle_frames, task_trace, envpool_infos
    )


def main() -> None:
    """Generate side-by-side MyoSuite render sample images."""
    args = _parse_args()
    traces = None
    if args.oracle_trace is not None:
        traces = json.loads(args.oracle_trace.read_text())["tasks"]
    if args.all_tasks:
        if args.out_dir is None:
            raise ValueError("--all_tasks requires --out_dir")
        task_order = []
        for task in _registered_tasks():
            task_id = str(task["id"])
            if traces is not None and task_id not in traces:
                continue
            if traces is None and bool(task["oracle_numpy2_broken"]):
                continue
            task_order.append(task_id)
        groups: dict[str, list[str]] = {
            "myobase": [],
            "myochallenge": [],
            "myodm": [],
        }
        for task_id in task_order:
            groups[_group_name(task_id)].append(task_id)
        stats = []
        for group, group_task_ids in groups.items():
            if traces is None:
                stats.extend(
                    _render_group_from_official(
                        group,
                        group_task_ids,
                        args.out_dir,
                        args.width,
                        args.height,
                        args.seed,
                        args.camera_id,
                    )
                )
            else:
                stats.extend(
                    _render_group(
                        group,
                        group_task_ids,
                        traces,
                        args.out_dir,
                        args.width,
                        args.height,
                        args.seed,
                    )
                )
    else:
        if args.out is None:
            raise ValueError("--out is required unless --all_tasks is set")
        if traces is None:
            stats = [
                _render_single_from_official(
                    args.task_id,
                    args.out,
                    args.width,
                    args.height,
                    args.seed,
                    args.camera_id,
                )
            ]
        else:
            stats = [
                _render_single(
                    args.task_id,
                    traces[args.task_id],
                    args.out,
                    args.width,
                    args.height,
                    args.seed,
                )
            ]
    if args.require_bitwise:
        failures = [stat for stat in stats if stat["nonzero_channels"] != 0]
        if failures:
            raise AssertionError(
                f"render comparison is not bitwise: {failures[:8]}"
            )
    if args.quiet:
        print(
            json.dumps(
                {
                    "max_abs_diff": max(
                        int(stat["max_abs_diff"]) for stat in stats
                    ),
                    "tasks": len(stats),
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps({"tasks": stats}, sort_keys=True))


if __name__ == "__main__":
    main()
