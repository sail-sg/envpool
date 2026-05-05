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
import os
import subprocess
import sys
import tempfile
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

_SYNC_STATE_KEYS = (
    "qpos0",
    "qvel0",
    "act0",
    "qacc0",
    "qacc_warmstart0",
    "ctrl",
    "site_pos",
    "site_quat",
    "site_size",
    "site_rgba",
    "body_pos",
    "body_quat",
    "body_mass",
    "geom_pos",
    "geom_quat",
    "geom_size",
    "geom_rgba",
    "geom_friction",
    "geom_aabb",
    "geom_rbound",
    "geom_contype",
    "geom_conaffinity",
    "geom_type",
    "geom_condim",
    "hfield_data",
    "mocap_pos",
    "mocap_quat",
    "fatigue_ma",
    "fatigue_mr",
    "fatigue_mf",
    "fatigue_tl",
)
_SYNC_STATE_SIZES = {
    "qpos0": "nq",
    "qvel0": "nv",
    "act0": "na",
    "qacc0": "nv",
    "qacc_warmstart0": "nv",
    "ctrl": "nu",
    "site_pos": "nsite3",
    "site_quat": "nsite4",
    "site_size": "nsite3",
    "site_rgba": "nsite4",
    "body_pos": "nbody3",
    "body_quat": "nbody4",
    "body_mass": "nbody",
    "geom_pos": "ngeom3",
    "geom_quat": "ngeom4",
    "geom_size": "ngeom3",
    "geom_rgba": "ngeom4",
    "geom_friction": "ngeom3",
    "geom_aabb": "ngeom6",
    "geom_rbound": "ngeom",
    "geom_contype": "ngeom",
    "geom_conaffinity": "ngeom",
    "geom_type": "ngeom",
    "geom_condim": "ngeom",
    "hfield_data": "nhfielddata",
    "mocap_pos": "nmocap3",
    "mocap_quat": "nmocap4",
    "fatigue_ma": "nu",
    "fatigue_mr": "nu",
    "fatigue_mf": "nu",
    "fatigue_tl": "nu",
}


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


def _envpool_frames_from_actions(
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


def _midpoint_action(env: object) -> np.ndarray:
    action_space = env.action_space  # type: ignore[attr-defined]
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    return ((low + high) * 0.5).astype(np.float32)


def _envpool_trace_record(
    task_id: str,
    width: int,
    height: int,
    seed: int,
    steps: int = 3,
) -> tuple[list[np.ndarray], list[Mapping[str, object]], dict[str, object]]:
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
        actions: list[list[float]] = []
        reset_before_step: list[bool] = []
        action = _midpoint_action(env)
        for _ in range(steps):
            actions.append(action.tolist())
            *_, info = env.step(action[None, :])
            frames.append(env.render()[0])
            infos.append(info)
            elapsed_step = int(np.asarray(info["elapsed_step"]).ravel()[0])
            reset_before_step.append(elapsed_step == 0)
        plan = {
            "actions": actions,
            "reset_before_step": reset_before_step,
            "sync_states": [_sync_state_from_info(item) for item in infos],
        }
        return frames, infos, plan
    finally:
        env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle_trace", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--all_tasks", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug_json", type=Path)
    parser.add_argument("--task_id", default="myoFingerReachFixed-v0")
    parser.add_argument("--seed", default=3, type=int)
    parser.add_argument("--width", default=160, type=int)
    parser.add_argument("--height", default=120, type=int)
    parser.add_argument("--camera_id", default=-1, type=int)
    return parser.parse_args()


def _runfiles_root() -> Path:
    path = Path(__file__).absolute()
    for parent in (path, *path.parents):
        if parent.name.endswith(".runfiles"):
            return parent
    runfiles_dir = os.environ.get("RUNFILES_DIR")
    if runfiles_dir:
        return Path(runfiles_dir)
    if "TEST_SRCDIR" in os.environ:
        return Path(os.environ["TEST_SRCDIR"])
    return path.parents[3]


def _oracle_probe_path() -> Path:
    runfiles = _runfiles_root()
    workspace = os.environ.get("TEST_WORKSPACE", "envpool")
    launcher_names = (
        ("myosuite_oracle_probe.exe", "myosuite_oracle_probe")
        if sys.platform == "win32"
        else ("myosuite_oracle_probe", "myosuite_oracle_probe.exe")
    )
    candidates = [
        runfiles / workspace / "envpool/mujoco" / launcher
        for launcher in launcher_names
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for launcher in launcher_names:
        matches = list(runfiles.rglob(launcher))
        if matches:
            return matches[0]
    raise RuntimeError(
        f"could not locate myosuite_oracle_probe under {runfiles}"
    )


def _sync_state_from_info(info: Mapping[str, object]) -> dict[str, Any]:
    dims = {
        "nq": int(np.asarray(info["model_nq"]).ravel()[0]),
        "nv": int(np.asarray(info["model_nv"]).ravel()[0]),
        "na": int(np.asarray(info["model_na"]).ravel()[0]),
        "nu": int(np.asarray(info["model_nu"]).ravel()[0]),
        "nsite": int(np.asarray(info["model_nsite"]).ravel()[0]),
        "nbody": int(np.asarray(info["model_nbody"]).ravel()[0]),
        "ngeom": int(np.asarray(info["model_ngeom"]).ravel()[0]),
        "nhfielddata": int(np.asarray(info["model_nhfielddata"]).ravel()[0]),
        "nmocap": int(np.asarray(info["model_nmocap"]).ravel()[0]),
    }
    dims.update({
        "nsite3": dims["nsite"] * 3,
        "nsite4": dims["nsite"] * 4,
        "nbody3": dims["nbody"] * 3,
        "nbody4": dims["nbody"] * 4,
        "ngeom3": dims["ngeom"] * 3,
        "ngeom4": dims["ngeom"] * 4,
        "ngeom6": dims["ngeom"] * 6,
        "nmocap3": dims["nmocap"] * 3,
        "nmocap4": dims["nmocap"] * 4,
    })
    sync_state = {}
    for key in _SYNC_STATE_KEYS:
        if key not in info:
            continue
        size = dims[_SYNC_STATE_SIZES[key]]
        sync_state[key] = (
            np.asarray(info[key][0], dtype=np.float64).ravel()[:size].tolist()
        )
    return sync_state


def _oracle_trace(
    task_ids: Sequence[str],
    trace_plan: Mapping[str, Mapping[str, object]] | None,
    width: int,
    height: int,
    seed: int,
    camera_id: int,
) -> Mapping[str, Mapping[str, object]]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out:
        out_path = Path(out.name)
    plan_path: Path | None = None
    cmd = [
        str(_oracle_probe_path()),
        "--mode",
        "trace",
        "--render",
        "--render_width",
        str(width),
        "--render_height",
        str(height),
        "--camera_id",
        str(camera_id),
        "--action_mode",
        "midpoint",
        "--steps",
        "3",
        "--seed",
        str(seed),
        "--out",
        str(out_path),
    ]
    if trace_plan is not None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as plan:
            plan_path = Path(plan.name)
        plan_path.write_text(json.dumps(trace_plan, sort_keys=True))
        cmd.extend(["--trace_plan", str(plan_path)])
    for task_id in task_ids:
        cmd.extend(["--task_id", task_id])
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
        return json.loads(out_path.read_text())["tasks"]
    finally:
        out_path.unlink(missing_ok=True)
        if plan_path is not None:
            plan_path.unlink(missing_ok=True)


def _task_frames_from_trace(
    task_id: str,
    task_trace: Mapping[str, object],
    width: int,
    height: int,
    seed: int,
    envpool_records: Mapping[
        str, tuple[list[np.ndarray], list[Mapping[str, object]]]
    ]
    | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[Mapping[str, object]]]:
    oracle_frames = [
        np.asarray(frame, dtype=np.uint8)
        for frame in task_trace["frames"]  # type: ignore[index]
    ]
    if envpool_records is not None and task_id in envpool_records:
        envpool_frames, envpool_infos = envpool_records[task_id]
    else:
        envpool_frames, envpool_infos = _envpool_frames_from_actions(
            task_id,
            width,
            height,
            seed,
            task_trace["actions"],  # type: ignore[arg-type,index]
        )
    if len(envpool_frames) != 4 or len(oracle_frames) != 4:
        raise ValueError("expected reset plus three step frames")
    return envpool_frames, oracle_frames, envpool_infos


def _render_stats(
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
    frame_diffs = []
    for idx, diff in enumerate(diffs):
        mismatched_pixels = int(np.count_nonzero(np.any(diff != 0, axis=-1)))
        frame_diffs.append(
            {
                "max_abs_diff": int(diff.max()),
                "mismatched_pixels": mismatched_pixels,
                "mean_abs_diff": float(np.mean(diff)),
                "first_diff": (
                    {
                        "index": [int(item) for item in np.argwhere(diff)[0]],
                        "envpool": [
                            int(item)
                            for item in envpool_frames[idx][
                                tuple(np.argwhere(diff)[0][:2])
                            ]
                        ],
                        "official": [
                            int(item)
                            for item in oracle_frames[idx][
                                tuple(np.argwhere(diff)[0][:2])
                            ]
                        ],
                    }
                    if mismatched_pixels
                    else None
                ),
            }
        )
    stats: dict[str, object] = {
        "task_id": task_id,
        "frames": len(diffs),
        "frame_diffs": frame_diffs,
        "max_abs_diff": max(int(diff.max()) for diff in diffs),
        "mismatched_pixels": sum(
            int(np.count_nonzero(np.any(diff != 0, axis=-1)))
            for diff in diffs
        ),
        "mean_abs_diff": max(float(np.mean(diff)) for diff in diffs),
        "envpool_elapsed_steps": [
            int(np.asarray(info["elapsed_step"]).ravel()[0])
            for info in envpool_infos
            if "elapsed_step" in info
        ],
        "envpool_times": [
            float(np.asarray(info["time"]).ravel()[0])
            for info in envpool_infos
            if "time" in info
        ],
    }
    states = [task_trace.get("reset_state", {})] + list(
        task_trace.get("states", [])  # type: ignore[arg-type]
    )
    state_diffs: dict[str, list[float]] = {}
    max_state_diff: dict[str, tuple[float, int]] = {}
    for step_id, (info, state) in enumerate(
        zip(envpool_infos, states, strict=False)
    ):
        if not isinstance(state, Mapping) or "qpos" not in state:
            continue
        for key in (
            "qpos",
            "qvel",
            "act",
            "actuator_force",
            "actuator_length",
            "actuator_velocity",
            "ctrl",
            "geom_rgba",
            "geom_xpos",
            "geom_xmat",
            "qacc_warmstart",
            "site_pos",
            "site_quat",
            "site_size",
            "site_xpos",
            "site_rgba",
            "body_pos",
            "body_quat",
            "light_xpos",
            "light_xdir",
            "mocap_pos",
            "mocap_quat",
            "fatigue_ma",
            "fatigue_mr",
            "fatigue_mf",
            "fatigue_tl",
            "fatigue_tauact",
            "fatigue_taudeact",
        ):
            if key not in info or key not in state:
                continue
            value = np.asarray(info[key], dtype=np.float64).ravel()
            oracle_value = np.asarray(state[key], dtype=np.float64).ravel()
            if oracle_value.size == 0:
                continue
            max_diff = float(
                np.max(np.abs(value[: oracle_value.size] - oracle_value))
            )
            state_diffs.setdefault(key, []).append(max_diff)
            if key not in max_state_diff or max_diff > max_state_diff[key][0]:
                max_state_diff[key] = (max_diff, step_id)
    for key, diffs_for_key in state_diffs.items():
        stats[f"max_{key}_abs_diff"] = max(diffs_for_key)
        stats[f"max_{key}_abs_diff_step"] = max_state_diff[key][1]
    return stats


def _render_single(
    task_id: str,
    task_trace: Mapping[str, object],
    out: Path,
    width: int,
    height: int,
    seed: int,
    envpool_records: Mapping[
        str, tuple[list[np.ndarray], list[Mapping[str, object]]]
    ]
    | None = None,
) -> dict[str, object]:
    envpool_frames, oracle_frames, envpool_infos = _task_frames_from_trace(
        task_id, task_trace, width, height, seed, envpool_records
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
    return _render_stats(
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
    envpool_records: Mapping[
        str, tuple[list[np.ndarray], list[Mapping[str, object]]]
    ],
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
            task_id, traces[task_id], width, height, seed, envpool_records
        )
        stats.append(
            _render_stats(
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


def main() -> None:
    """Generate side-by-side MyoSuite render sample images."""
    args = _parse_args()
    traces = None
    envpool_records: dict[
        str, tuple[list[np.ndarray], list[Mapping[str, object]]]
    ] = {}
    trace_plan: dict[str, Mapping[str, object]] = {}
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
        if traces is None:
            for task_id in task_order:
                frames, infos, plan = _envpool_trace_record(
                    task_id, args.width, args.height, args.seed
                )
                envpool_records[task_id] = (frames, infos)
                trace_plan[task_id] = plan
            traces = _oracle_trace(
                task_order,
                trace_plan,
                args.width,
                args.height,
                args.seed,
                args.camera_id,
            )
        groups: dict[str, list[str]] = {
            "myobase": [],
            "myochallenge": [],
            "myodm": [],
        }
        for task_id in task_order:
            groups[_group_name(task_id)].append(task_id)
        stats = []
        for group, group_task_ids in groups.items():
            stats.extend(
                _render_group(
                    group,
                    group_task_ids,
                    traces,
                    envpool_records,
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
            frames, infos, plan = _envpool_trace_record(
                args.task_id, args.width, args.height, args.seed
            )
            envpool_records[args.task_id] = (frames, infos)
            trace_plan[args.task_id] = plan
            traces = _oracle_trace(
                (args.task_id,),
                trace_plan,
                args.width,
                args.height,
                args.seed,
                args.camera_id,
            )
        stats = [
            _render_single(
                args.task_id,
                traces[args.task_id],
                args.out,
                args.width,
                args.height,
                args.seed,
                envpool_records,
            )
        ]
    if args.debug_json is not None:
        args.debug_json.write_text(
            json.dumps(
                {
                    "stats": stats,
                    "trace_plan": trace_plan,
                    "traces": traces,
                },
                sort_keys=True,
            )
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
