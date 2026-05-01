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
"""Generate the MyoSuite EnvPool-vs-official render sample for docs."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _bootstrap_envpool_namespace() -> None:
    """Load envpool modules without importing envpool.entry."""
    if "envpool" in sys.modules:
        return
    for path in sys.path:
        envpool_root = Path(path) / "envpool"
        if (envpool_root / "registration.py").is_file():
            module = types.ModuleType("envpool")
            module.__file__ = str(envpool_root / "__init__.py")
            module.__path__ = [str(envpool_root)]  # type: ignore[attr-defined]
            sys.modules["envpool"] = module
            return
    raise RuntimeError("could not locate envpool package on PYTHONPATH")


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


def _envpool_frames(
    task_id: str, width: int, height: int, seed: int, actions: list[list[float]]
) -> list[np.ndarray]:
    env = _make_gymnasium(
        task_id,
        num_envs=1,
        seed=seed,
        render_mode="rgb_array",
        render_width=width,
        render_height=height,
    )
    try:
        env.reset()
        frames = [env.render()[0]]
        for action in actions:
            env.step(np.asarray(action, dtype=np.float32)[None, :])
            frames.append(env.render()[0])
        return frames
    finally:
        env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle_trace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--task_id", default="myoFingerReachFixed-v0")
    parser.add_argument("--seed", default=3, type=int)
    parser.add_argument("--width", default=160, type=int)
    parser.add_argument("--height", default=120, type=int)
    return parser.parse_args()


def main() -> None:
    """Generate the side-by-side MyoSuite render sample image."""
    args = _parse_args()
    trace = json.loads(args.oracle_trace.read_text())
    task_trace = trace["tasks"][args.task_id]
    oracle_frames = [
        np.asarray(frame, dtype=np.uint8) for frame in task_trace["frames"]
    ]
    envpool_frames = _envpool_frames(
        args.task_id,
        args.width,
        args.height,
        args.seed,
        task_trace["actions"],
    )
    if len(envpool_frames) != 4 or len(oracle_frames) != 4:
        raise ValueError("expected reset plus three step frames")

    margin = 18
    label_h = 20
    gutter = 12
    row_gap = 10
    cell_w = args.width
    cell_h = args.height + label_h
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)

    diffs = [
        np.abs(a.astype(np.int16) - b.astype(np.int16))
        for a, b in zip(envpool_frames, oracle_frames, strict=True)
    ]
    stats = {
        "task_id": args.task_id,
        "frames": len(diffs),
        "max_abs_diff": max(int(diff.max()) for diff in diffs),
        "nonzero_channels": sum(int(np.count_nonzero(diff)) for diff in diffs),
    }
    print(json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
