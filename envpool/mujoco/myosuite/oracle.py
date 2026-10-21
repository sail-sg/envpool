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
"""Native hidden-state transfer for MyoSuite validation and doc images."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def oracle_probe_cmd() -> list[str]:
    """Locate the Bazel probe launcher, including Windows manifest layouts."""
    path = _oracle_probe_path()
    if sys.platform == "win32" and path.suffix.lower() != ".exe":
        return [sys.executable, str(path)]
    return [str(path)]


def _oracle_probe_path() -> Path:
    runfiles = Path(os.environ["TEST_SRCDIR"])
    workspace = os.environ.get("TEST_WORKSPACE", "envpool")
    launcher_names: tuple[str, ...] = (
        "myosuite_oracle_probe",
        "myosuite_oracle_probe.exe",
    )
    logical_suffixes = (
        tuple(f"envpool/mujoco/{launcher}" for launcher in launcher_names)
        + launcher_names
    )
    manifest = os.environ.get("RUNFILES_MANIFEST_FILE")
    if manifest:
        with Path(manifest).open(encoding="utf-8") as f:
            for line in f:
                logical, _, physical = line.rstrip("\n").partition(" ")
                logical = logical.replace("\\", "/")
                if any(logical.endswith(suffix) for suffix in logical_suffixes):
                    candidate = Path(physical or logical)
                    if candidate.is_file():
                        return candidate
    candidates = [
        runfiles / workspace / "envpool/mujoco" / launcher
        for launcher in launcher_names
    ]
    if sys.platform == "win32":
        candidates.extend(
            runfiles.parent / launcher for launcher in launcher_names
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for launcher in launcher_names:
        for match in runfiles.rglob(launcher):
            if match.is_file():
                return match
    raise RuntimeError(
        f"could not locate myosuite_oracle_probe under {runfiles}"
    )


_SYNC_STATE_SIZES: dict[str, tuple[str | None, int]] = {
    "target_jnt_value": ("nq", 1),
    "baoding_goal_parameters": (None, 5),
    "chase_task": (None, 1),
    "npc_parameters": (None, 3),
    "qpos0": ("nq", 1),
    "qvel0": ("nv", 1),
    "act0": ("na", 1),
    "qacc0": ("nv", 1),
    "qacc_warmstart0": ("nv", 1),
    "ctrl": ("nu", 1),
    "site_pos": ("nsite", 3),
    "site_quat": ("nsite", 4),
    "site_size": ("nsite", 3),
    "site_rgba": ("nsite", 4),
    "body_pos": ("nbody", 3),
    "body_quat": ("nbody", 4),
    "body_mass": ("nbody", 1),
    "geom_pos": ("ngeom", 3),
    "geom_quat": ("ngeom", 4),
    "geom_size": ("ngeom", 3),
    "geom_rgba": ("ngeom", 4),
    "geom_friction": ("ngeom", 3),
    "geom_aabb": ("ngeom", 6),
    "geom_rbound": ("ngeom", 1),
    "geom_contype": ("ngeom", 1),
    "geom_conaffinity": ("ngeom", 1),
    "geom_type": ("ngeom", 1),
    "geom_condim": ("ngeom", 1),
    "mesh_polymap": ("nmeshpolymap", 1),
    "mesh_polynormal": ("nmeshpoly", 3),
    "mesh_polyvert": ("nmeshpolyvert", 1),
    "mesh_polyvertadr": ("nmeshpoly", 1),
    "mesh_polyvertnum": ("nmeshpoly", 1),
    "hfield_data": ("nhfielddata", 1),
    "mocap_pos": ("nmocap", 3),
    "mocap_quat": ("nmocap", 4),
    "fatigue_ma": ("nu", 1),
    "fatigue_mr": ("nu", 1),
    "fatigue_mf": ("nu", 1),
    "fatigue_tl": ("nu", 1),
}


def sync_state_from_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Trim native reset state arrays for the official oracle."""
    sync_state = {}
    for key, (dimension, width) in _SYNC_STATE_SIZES.items():
        if key not in info:
            continue
        count = (
            int(np.asarray(info[f"model_{dimension}"]).ravel()[0])
            if dimension is not None
            else 1
        )
        size = count * width
        sync_state[key] = np.asarray(info[key][0]).ravel()[:size].tolist()
    sampled = sampled_state_from_info(info)
    sampled.pop("npc_velocity", None)  # Reset precedes the first NPC command.
    return dict(sync_state, **sampled)


def sampled_state_from_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Export sampled goals/commands, without copying evolved physics state."""
    state = {}
    if int(info["npc_parameters"][0, 0]) in (2, 3):
        state["npc_velocity"] = info["npc_velocity"][0].tolist()
    reference = info["random_reference"][0]
    if reference.size > 1:
        state["random_reference"] = reference.tolist()
    return state


def physics_state_from_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Current native integrator state, before the next transition is tested."""
    return {
        key: np
        .asarray(info[key][0])
        .ravel()[: int(np.asarray(info[dimension]).ravel()[0])]
        .tolist()
        for key, dimension in (
            ("qpos", "model_nq"),
            ("qvel", "model_nv"),
            ("act", "model_na"),
            ("qacc_warmstart", "model_nv"),
        )
    }
