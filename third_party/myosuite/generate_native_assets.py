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

"""Generate native MyoSuite assets from pinned upstream archives.

This tool runs at build/codegen time only. It imports the pinned upstream
MyoSuite source as an oracle to derive compact registry and task metadata, then
emits native C++/Python data files consumed by EnvPool. The generated files are
not checked in, and the native runtime never imports the official Python
package.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import os
import posixpath
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_reference_data
import generate_task_metadata
import generate_task_registry

_ORACLE_VERSION = generate_task_registry.ORACLE_VERSION
_ORACLE_COMMIT = generate_task_registry.ORACLE_COMMIT
_BROKEN_IDS = set(generate_task_registry.BROKEN_IDS)
_SIMHIVE_DIRS = {
    "mpl": "MPL_sim",
    "ycb": "YCB_sim",
    "furniture": "furniture_sim",
    "myo": "myo_sim",
    "object": "object_sim",
}
_SIMHIVE_REPOS = {
    "mpl": "myosuite_mpl_sim",
    "ycb": "myosuite_ycb_sim",
    "furniture": "myosuite_furniture_sim",
    "myo": "myosuite_myo_sim",
    "object": "myosuite_object_sim",
}
_WINDOWS_SHORT_IMPORT_PACKAGES = ("mujoco", "h5py")
_DLL_DIRECTORY_HANDLES: list[Any] = []


def _manifest_paths(path: Path) -> list[Path]:
    return [
        Path(line) for line in path.read_text().splitlines() if line.strip()
    ]


def _common_root(paths: list[Path]) -> Path:
    if not paths:
        raise ValueError("empty source manifest")
    return Path(os.path.commonpath([str(path) for path in paths]))


def _repo_root(paths: list[Path], repo: str) -> Path:
    for path in paths:
        parts = path.parts
        if repo in parts:
            idx = parts.index(repo)
            return Path(*parts[: idx + 1])
    return _common_root(paths)


def _myosuite_source_root(paths: list[Path]) -> Path:
    repo_root = _repo_root(paths, "myosuite_source")
    if (repo_root / "myosuite").is_dir():
        return repo_root
    root = _common_root(paths)
    if root.name == "myosuite":
        return root.parent
    package = root / "myosuite"
    if package.is_dir():
        return root
    for path in paths:
        parts = path.parts
        if "myosuite" in parts:
            idx = parts.index("myosuite")
            return Path(*parts[:idx])
    raise ValueError(f"could not infer MyoSuite source root from {root}")


def _patch_codegen_only_imports(package: Path) -> None:
    import_utils = package / "utils" / "import_utils.py"
    text = import_utils.read_text()
    eager_git = "from os.path import expanduser\nimport git\n\n\n"
    fetch_def = (
        "def fetch_git(repo_url, commit_hash, clone_directory, "
        "clone_path=None):\n"
    )
    lazy_fetch = fetch_def + "    import git\n"
    if eager_git not in text:
        if lazy_fetch in text:
            return
        raise ValueError("unexpected MyoSuite import_utils.py layout")
    text = text.replace(eager_git, "from os.path import expanduser\n\n\n", 1)
    if fetch_def not in text:
        raise ValueError("unexpected MyoSuite fetch_git layout")
    text = text.replace(fetch_def, lazy_fetch, 1)
    if import_utils.is_symlink():
        import_utils.unlink()
    import_utils.write_text(text)


@contextlib.contextmanager
def _assembled_source(
    source_manifest: Path, sim_manifests: dict[str, Path]
) -> Any:
    source_root = _myosuite_source_root(_manifest_paths(source_manifest))
    sim_roots = {
        key: _repo_root(_manifest_paths(manifest), _SIMHIVE_REPOS[key])
        for key, manifest in sim_manifests.items()
    }
    with tempfile.TemporaryDirectory(
        prefix="envpool-myosuite-src-",
        ignore_cleanup_errors=os.name == "nt",
    ) as tmp:
        root = Path(tmp)
        package = root / "myosuite"
        shutil.copytree(source_root / "myosuite", package, symlinks=True)
        _patch_codegen_only_imports(package)
        simhive = package / "simhive"
        simhive.mkdir(exist_ok=True)
        for key, dirname in _SIMHIVE_DIRS.items():
            dst = simhive / dirname
            if dst.exists() or dst.is_symlink():
                if dst.is_dir() and not dst.is_symlink():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            shutil.copytree(sim_roots[key], dst, symlinks=True)
        old_path = list(sys.path)
        for name in tuple(sys.modules):
            if name == "myosuite" or name.startswith("myosuite."):
                del sys.modules[name]
        sys.path.insert(0, str(root))
        try:
            yield root
        finally:
            sys.path[:] = old_path
            for name in tuple(sys.modules):
                if name == "myosuite" or name.startswith("myosuite."):
                    del sys.modules[name]


def _import_official() -> tuple[Any, Any, Any]:
    official_myosuite = importlib.import_module("myosuite")
    from myosuite import gym_registry_specs
    from myosuite.utils import gym

    return official_myosuite, gym_registry_specs, gym


def _copy_short_import_package(
    source_root: Path, package_name: str
) -> list[Path]:
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.submodule_search_locations is None:
        return []
    source = Path(next(iter(spec.submodule_search_locations)))
    if not source.is_dir():
        return []
    destination_root = source_root / "_oracle_site"
    destination = destination_root / package_name
    if not destination.exists():
        shutil.copytree(
            source,
            destination,
            symlinks=False,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    copied = [destination]
    sibling = source.parent / f"{package_name}.libs"
    if sibling.is_dir():
        sibling_destination = destination_root / sibling.name
        if not sibling_destination.exists():
            shutil.copytree(sibling, sibling_destination, symlinks=False)
        copied.append(sibling_destination)
    return copied


def _shorten_windows_binary_imports(source_root: Path) -> None:
    if os.name != "nt":
        return
    destination_root = source_root / "_oracle_site"
    copied: list[Path] = []
    for package_name in _WINDOWS_SHORT_IMPORT_PACKAGES:
        copied.extend(_copy_short_import_package(source_root, package_name))
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                del sys.modules[name]
    if copied:
        sys.path.insert(0, str(destination_root))
    if hasattr(os, "add_dll_directory"):
        for path in copied:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    if array.dtype == object:
        return [str(item) for item in array.ravel()]
    return array.tolist()


def _names_from_ids(model: Any, obj_type: Any, ids: list[int]) -> list[str]:
    import mujoco

    raw_model = model.ptr if hasattr(model, "ptr") else model
    return [
        mujoco.mj_id2name(raw_model, int(obj_type), int(obj_id))
        for obj_id in ids
    ]


def _state_report(env: Any) -> dict[str, Any]:
    model = env.sim.model
    data = env.sim.data
    return {
        "act": _jsonable(data.act) if model.na > 0 else [],
        "ctrl": _jsonable(data.ctrl),
        "qacc_warmstart": _jsonable(data.qacc_warmstart),
        "body_pos": _jsonable(model.body_pos),
        "body_quat": _jsonable(model.body_quat),
        "mocap_pos": _jsonable(data.mocap_pos),
        "mocap_quat": _jsonable(data.mocap_quat),
        "qpos": _jsonable(data.qpos),
        "qvel": _jsonable(data.qvel),
        "site_pos": _jsonable(model.site_pos),
        "site_quat": _jsonable(model.site_quat),
        "time": float(data.time),
    }


def _metadata_report(task_ids: list[str], gym: Any) -> dict[str, Any]:
    import mujoco

    tasks: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        env = gym.make(task_id)
        try:
            unwrapped = env.unwrapped
            model = unwrapped.sim.model
            data = unwrapped.sim.data
            task: dict[str, Any] = {
                "action_shape": list(env.action_space.shape),
                "entry_class": type(unwrapped).__name__,
                "frame_skip": int(unwrapped.frame_skip),
                "init_qpos": _jsonable(unwrapped.init_qpos),
                "init_qvel": _jsonable(unwrapped.init_qvel),
                "model_nq": int(model.nq),
                "model_nv": int(model.nv),
                "model_na": int(model.na),
                "model_nu": int(model.nu),
                "obs_keys": list(unwrapped.obs_keys),
                "observation_shape": list(env.observation_space.shape),
                "rwd_keys_wt": dict(unwrapped.rwd_keys_wt),
            }
            for attr in (
                "far_th",
                "goal_th",
                "hip_period",
                "max_rot",
                "min_height",
                "normalize_act",
                "pose_thd",
                "reset_type",
                "target_rot",
                "target_x_vel",
                "target_y_vel",
                "terrain",
                "variant",
            ):
                if hasattr(unwrapped, attr):
                    task[attr] = _jsonable(getattr(unwrapped, attr))
            if hasattr(unwrapped, "tip_sids"):
                task["tip_sites"] = _names_from_ids(
                    model, mujoco.mjtObj.mjOBJ_SITE, unwrapped.tip_sids
                )
            if hasattr(unwrapped, "target_sids"):
                task["target_sites"] = _names_from_ids(
                    model, mujoco.mjtObj.mjOBJ_SITE, unwrapped.target_sids
                )
            if hasattr(unwrapped, "target_jnt_ids"):
                task["target_joints"] = _names_from_ids(
                    model, mujoco.mjtObj.mjOBJ_JOINT, unwrapped.target_jnt_ids
                )
            for attr in (
                "target_jnt_range",
                "target_jnt_value",
                "target_reach_range",
            ):
                if hasattr(unwrapped, attr):
                    task[attr] = _jsonable(getattr(unwrapped, attr))
            task["initial_state"] = {
                "qpos": _jsonable(data.qpos),
                "qvel": _jsonable(data.qvel),
                "act": _jsonable(data.act) if model.na > 0 else [],
                "qacc_warmstart": _jsonable(data.qacc_warmstart),
                "site_pos": _jsonable(model.site_pos),
                "site_quat": _jsonable(model.site_quat),
                "body_pos": _jsonable(model.body_pos),
                "body_quat": _jsonable(model.body_quat),
            }
            env.reset(seed=0)
            task["reset_state"] = _state_report(unwrapped)
            tasks[task_id] = task
        finally:
            env.close()
    return {"tasks": tasks, "version": _ORACLE_VERSION}


def _kind(entry_point: str, task_id: str) -> str:
    if "myodm_v0:TrackEnv" in entry_point:
        return "kMyoDmTrack"
    if "torso_v0" in entry_point:
        return "kTorsoPose"
    if "pose_v0" in entry_point:
        return "kPose"
    if "walk_v0:ReachEnvV0" in entry_point:
        return "kWalkReach"
    if "walk_v0:WalkEnvV0" in entry_point:
        return "kWalk"
    if "walk_v0:TerrainEnvV0" in entry_point:
        return "kTerrain"
    if "reach_v0" in entry_point:
        return "kReach"
    if "key_turn_v0" in entry_point:
        return "kKeyTurn"
    if "obj_hold_v0" in entry_point:
        return "kObjHoldRandom" if "Random" in task_id else "kObjHoldFixed"
    if "pen_v0" in entry_point:
        return "kPenTwirlRandom" if "Random" in task_id else "kPenTwirlFixed"
    if "reorient_sar_v0" in entry_point:
        return "kReorientSar"
    if "baoding" in entry_point:
        return "kChallengeBaoding"
    if "bimanual" in entry_point:
        return "kChallengeBimanual"
    if "chasetag" in entry_point:
        return "kChallengeChaseTag"
    if "relocate" in entry_point:
        return "kChallengeRelocate"
    if "reorient_v0" in entry_point:
        return "kChallengeReorient"
    if "run_track" in entry_point:
        return "kChallengeRunTrack"
    if "soccer" in entry_point:
        return "kChallengeSoccer"
    if "tabletennis" in entry_point:
        return "kChallengeTableTennis"
    raise ValueError(f"unknown MyoSuite task kind for {task_id}: {entry_point}")


def _muscle(kwargs: dict[str, Any]) -> str:
    value = kwargs.get("muscle_condition", "")
    return {
        "sarcopenia": "kSarcopenia",
        "fatigue": "kFatigue",
        "reafferentation": "kReafferentation",
    }.get(value, "kNormal")


def _normalize_path(value: str, source_root: Path) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        return value.lstrip("/")
    marker = "/myosuite/"
    raw = str(path)
    if marker in raw:
        return "myosuite/" + posixpath.normpath(raw.split(marker, 1)[1])
    try:
        rel = path.resolve().relative_to(source_root.resolve() / "myosuite")
        return "myosuite/" + rel.as_posix()
    except ValueError:
        return value


def _task_from_spec(
    task_id: str,
    spec: Any,
    source_root: Path,
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    kwargs = dict(getattr(spec, "kwargs", {}) or {})
    entry_point = str(spec.entry_point)
    object_name = str(kwargs.get("object_name", ""))
    reference_path = ""
    model_path = _normalize_path(str(kwargs.get("model_path", "")), source_root)
    if "ArmReach" in task_id:
        model_path = "myosuite/simhive/myo_sim/arm/myoarm_reach.xml"
    if _kind(entry_point, task_id) == "kChallengeTableTennis":
        model_path = (
            "myosuite/envs/myo/assets/arm/myoarm_tabletennis_native.xml"
        )
    if _kind(entry_point, task_id) == "kMyoDmTrack":
        model_path = (
            f"myosuite/envs/myo/assets/hand/myohand_object_{object_name}.xml"
        )
        reference = kwargs.get("reference")
        if isinstance(reference, str):
            reference_path = _normalize_path(reference, source_root)
    task = {
        "id": task_id,
        "entry_point": entry_point,
        "kind": _kind(entry_point, task_id),
        "model_path": model_path,
        "reference_path": reference_path,
        "object_name": object_name,
        "obs_dim": 0,
        "action_dim": 0,
        "max_episode_steps": int(spec.max_episode_steps),
        "frame_skip": int(kwargs.get("frame_skip", 10)),
        "normalize_act": bool(
            metadata.get(task_id, {}).get(
                "normalize_act", kwargs.get("normalize_act", False)
            )
        ),
        "muscle": _muscle(kwargs),
        "oracle_numpy2_broken": task_id in _BROKEN_IDS,
    }
    if task_id in metadata:
        task["obs_dim"] = int(metadata[task_id]["observation_shape"][0])
        task["action_dim"] = int(metadata[task_id]["action_shape"][0])
        task["frame_skip"] = int(metadata[task_id]["frame_skip"])
    else:
        raise ValueError(f"missing metadata for {task_id}")
    return task


def _write_outputs(args: argparse.Namespace, source_root: Path) -> None:
    _shorten_windows_binary_imports(source_root)
    official_myosuite, gym_registry_specs, gym = _import_official()
    if official_myosuite.__version__ != _ORACLE_VERSION:
        raise ValueError(
            f"expected MyoSuite {_ORACLE_VERSION}, "
            f"got {official_myosuite.__version__}"
        )
    task_ids = list(official_myosuite.myosuite_env_suite)
    metadata_report = _metadata_report(task_ids, gym)
    metadata = cast(dict[str, dict[str, Any]], metadata_report["tasks"])
    specs = gym_registry_specs()
    tasks = [
        _task_from_spec(task_id, specs[task_id], source_root, metadata)
        for task_id in task_ids
    ]

    generate_task_registry._write_json(tasks, args.out_tasks_json)
    generate_task_registry._write_header(tasks, args.out_tasks_header)

    metadata_entries = [
        generate_task_metadata._entry(task, metadata.get(task["id"]))
        for task in tasks
    ]
    args.out_metadata_json.write_text(
        json.dumps(metadata_entries, indent=2, sort_keys=True) + "\n"
    )
    generate_task_metadata._write_header(
        metadata_entries, args.out_metadata_header
    )
    args.out_oracle_json.write_text(
        json.dumps(
            {
                "commit": _ORACLE_COMMIT,
                "numpy2_broken_ids": sorted(_BROKEN_IDS),
                "version": _ORACLE_VERSION,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    reference_entries = [
        generate_reference_data._reference_entry(task, source_root)
        for task in tasks
        if task["kind"] == "kMyoDmTrack"
    ]
    generate_reference_data._write_header(
        reference_entries, args.out_reference_header
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--myosuite-manifest", type=Path, required=True)
    parser.add_argument("--mpl-manifest", type=Path, required=True)
    parser.add_argument("--ycb-manifest", type=Path, required=True)
    parser.add_argument("--furniture-manifest", type=Path, required=True)
    parser.add_argument("--myo-manifest", type=Path, required=True)
    parser.add_argument("--object-manifest", type=Path, required=True)
    parser.add_argument("--out-tasks-json", type=Path, required=True)
    parser.add_argument("--out-tasks-header", type=Path, required=True)
    parser.add_argument("--out-metadata-json", type=Path, required=True)
    parser.add_argument("--out-metadata-header", type=Path, required=True)
    parser.add_argument("--out-oracle-json", type=Path, required=True)
    parser.add_argument("--out-reference-header", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Generate all native MyoSuite registry and metadata assets."""
    os.environ.setdefault("ROBOHIVE_VERBOSITY", "SILENT")
    warnings.filterwarnings("ignore")
    args = _parse_args()
    sim_manifests = {
        "mpl": args.mpl_manifest,
        "ycb": args.ycb_manifest,
        "furniture": args.furniture_manifest,
        "myo": args.myo_manifest,
        "object": args.object_manifest,
    }
    with _assembled_source(args.myosuite_manifest, sim_manifests) as root:
        _write_outputs(args, root)


if __name__ == "__main__":
    main()
