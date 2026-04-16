#!/usr/bin/env python3
# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate canonical MyoSuite registry metadata from upstream source."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _load_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _keyword_string(node: ast.Call, arg_name: str) -> str | None:
    for keyword in node.keywords:
        if keyword.arg == arg_name:
            return _constant_string(keyword.value)
    return None


def _direct_ids_from_register_env_with_variants(path: Path) -> list[str]:
    ids: list[str] = []
    for node in ast.walk(_load_module(path)):
        if (
            isinstance(node, ast.Call)
            and _call_name(node) == "register_env_with_variants"
        ):
            env_id = _keyword_string(node, "id")
            if env_id is not None:
                ids.append(env_id)
    return ids


def _expand_upstream_variants(env_ids: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for env_id in env_ids:
        expanded.append(env_id)
        if env_id.startswith("myo"):
            expanded.append(env_id[:3] + "Sarc" + env_id[3:])
            expanded.append(env_id[:3] + "Fati" + env_id[3:])
        if env_id.startswith("myoHand"):
            expanded.append(env_id[:3] + "Reaf" + env_id[3:])
    return expanded


def _dedupe_sorted(values: Iterable[str]) -> list[str]:
    return sorted(set(values))


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_json_value(value[key]) for key in sorted(value)
        }
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _normalize_json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return getattr(value, "__name__", repr(value))
    return value


def _normalize_upstream_path(
    value: str, module_dir: Path, myosuite_root: Path
) -> str:
    candidate = Path(value)
    if candidate.is_absolute() and str(candidate).startswith(
        str(myosuite_root)
    ):
        resolved = candidate.resolve()
    else:
        relative = value[1:] if value.startswith("/") else value
        resolved = (module_dir / relative).resolve()
    if "myosuite" in resolved.parts:
        index = max(
            i for i, part in enumerate(resolved.parts) if part == "myosuite"
        )
        return Path(*resolved.parts[index + 1 :]).as_posix()
    return resolved.relative_to(myosuite_root).as_posix()


def _normalize_captured_kwargs(
    kwargs: dict[str, Any], module_dir: Path, myosuite_root: Path
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in sorted(kwargs):
        value = _normalize_json_value(kwargs[key])
        if isinstance(value, str) and (
            key.endswith("_path") or key == "reference"
        ):
            value = _normalize_upstream_path(value, module_dir, myosuite_root)
        normalized[key] = value
    return normalized


def _model_placeholders(model_path: str, myosuite_root: Path) -> list[str]:
    placeholders: list[str] = []
    xml_path = myosuite_root / model_path
    if not xml_path.exists():
        return placeholders
    xml_text = xml_path.read_text()
    if "OBJECT_NAME" in xml_text:
        placeholders.append("OBJECT_NAME")
    return placeholders


def _capture_suite_entries(
    upstream_root: Path, suite_name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    myosuite_root = upstream_root / "myosuite"
    init_path = myosuite_root / "envs" / "myo" / suite_name / "__init__.py"
    records: list[dict[str, Any]] = []
    variant_records: list[dict[str, Any]] = []

    def fake_register(
        *, id: str, entry_point: str, max_episode_steps: int, kwargs: Any
    ) -> None:
        records.append({
            "id": id,
            "entry_point": entry_point,
            "max_episode_steps": max_episode_steps,
            "kwargs": dict(kwargs),
        })

    def fake_register_env_variant(
        *,
        env_id: str,
        variants: dict[str, Any],
        variant_id: str,
        silent: bool = True,
    ) -> None:
        variant_records.append({
            "env_id": env_id,
            "variant_id": variant_id,
            "variants": _normalize_json_value(variants),
            "silent": silent,
        })

    class FakeBackend:
        @staticmethod
        def get_sim_backend() -> str:
            return "mujoco"

    module_names = (
        "mujoco",
        "myosuite",
        "myosuite.utils",
        "myosuite.envs",
        "myosuite.envs.myo",
        "myosuite.envs.myo.myobase",
        "myosuite.envs.env_variants",
        "myosuite.physics",
        "myosuite.physics.sim_scene",
    )
    originals = {name: sys.modules.get(name) for name in module_names}
    modules = {name: types.ModuleType(name) for name in module_names}
    modules["mujoco"].MjSpec = type("MjSpec", (), {})
    modules["mujoco"].mjtGeom = types.SimpleNamespace(
        mjGEOM_MESH=0,
        mjGEOM_SPHERE=1,
    )
    modules["myosuite.utils"].gym = types.SimpleNamespace(
        register=fake_register
    )
    modules[
        "myosuite.envs.env_variants"
    ].register_env_variant = fake_register_env_variant

    def register_env_with_variants(
        id: str, entry_point: str, max_episode_steps: int, kwargs: Any
    ) -> None:
        fake_register(
            id=id,
            entry_point=entry_point,
            max_episode_steps=max_episode_steps,
            kwargs=kwargs,
        )
        if id.startswith("myo"):
            fake_register_env_variant(
                env_id=id,
                variants={"muscle_condition": "sarcopenia"},
                variant_id=id[:3] + "Sarc" + id[3:],
                silent=True,
            )
            fake_register_env_variant(
                env_id=id,
                variants={"muscle_condition": "fatigue"},
                variant_id=id[:3] + "Fati" + id[3:],
                silent=True,
            )
        if id.startswith("myoHand"):
            fake_register_env_variant(
                env_id=id,
                variants={"muscle_condition": "reafferentation"},
                variant_id=id[:3] + "Reaf" + id[3:],
                silent=True,
            )

    modules[
        "myosuite.envs.myo.myobase"
    ].register_env_with_variants = register_env_with_variants
    modules["myosuite.physics.sim_scene"].SimBackend = FakeBackend
    modules["myosuite"].utils = modules["myosuite.utils"]
    modules["myosuite"].envs = modules["myosuite.envs"]
    modules["myosuite"].physics = modules["myosuite.physics"]
    modules["myosuite.envs"].myo = modules["myosuite.envs.myo"]
    modules["myosuite.envs.myo"].myobase = modules["myosuite.envs.myo.myobase"]
    modules["myosuite.envs"].env_variants = modules[
        "myosuite.envs.env_variants"
    ]
    modules["myosuite.physics"].sim_scene = modules[
        "myosuite.physics.sim_scene"
    ]

    try:
        for name, module in modules.items():
            sys.modules[name] = module
        namespace = {
            "__file__": str(init_path),
            "__name__": f"_capture_{suite_name}",
        }
        exec(compile(init_path.read_text(), str(init_path), "exec"), namespace)
    finally:
        for name, module in originals.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return records, variant_records


def _captured_entry_to_metadata(
    record: dict[str, Any],
    logical_suite_name: str,
    registration_suite_name: str,
    upstream_root: Path,
    variant_map: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    myosuite_root = upstream_root / "myosuite"
    entry_module, class_name = record["entry_point"].split(":")
    module_path = upstream_root / Path(*entry_module.split("."))
    module_path = module_path.with_suffix(".py")
    kwargs = _normalize_captured_kwargs(
        record["kwargs"], module_path.parent, myosuite_root
    )
    model_path = kwargs["model_path"]
    return {
        "class_name": class_name,
        "entry_module": entry_module,
        "id": record["id"],
        "kwargs": kwargs,
        "max_episode_steps": record["max_episode_steps"],
        "model_placeholders": _model_placeholders(model_path, myosuite_root),
        "registration_suite": registration_suite_name,
        "suite": logical_suite_name,
        "variant_defs": sorted(
            variant_map.get(record["id"], []),
            key=lambda item: item["variant_id"],
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _copy_or_symlink_tree(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def _find_staged_runtime_asset_root() -> Path | None:
    candidates: list[Path] = []
    for root in (
        Path.cwd(),
        _repo_root(),
        Path(os.environ["RUNFILES_DIR"])
        if os.environ.get("RUNFILES_DIR")
        else None,
        Path(os.environ["TEST_SRCDIR"])
        if os.environ.get("TEST_SRCDIR")
        else None,
    ):
        if root is None:
            continue
        candidates.extend(
            [
                root / "envpool/mujoco/myosuite_assets",
                root / "envpool/envpool/mujoco/myosuite_assets",
                root / "mujoco/myosuite_assets",
            ]
        )
    required_runtime_asset = (
        Path("envs") / "myo" / "assets" / "hand" / "myohand_object.xml"
    )
    for candidate in candidates:
        if (candidate / required_runtime_asset).exists():
            return candidate
    return None


@contextmanager
def _staged_runtime_asset_base(upstream_root: Path) -> Iterable[Path]:
    external_root = upstream_root.parent
    sibling_roots = {
        "myo_sim": external_root / "myo_sim_src",
        "furniture_sim": external_root / "furniture_sim_src",
        "object_sim": external_root / "object_sim_src",
        "mpl_sim": external_root / "mpl_sim_src",
        "ycb_sim": external_root / "ycb_sim_src",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        asset_root = base_path / "mujoco" / "myosuite_assets"
        if all(path.exists() for path in sibling_roots.values()):
            mappings = {
                upstream_root / "myosuite" / "envs": asset_root / "envs",
                sibling_roots["myo_sim"]: asset_root / "simhive/myo_sim",
                sibling_roots["furniture_sim"]: asset_root
                / "simhive/furniture_sim",
                sibling_roots["object_sim"]: asset_root / "simhive/object_sim",
                sibling_roots["mpl_sim"]: asset_root / "simhive/MPL_sim",
                sibling_roots["ycb_sim"]: asset_root / "simhive/YCB_sim",
            }
            for src, dst in mappings.items():
                dst.parent.mkdir(parents=True, exist_ok=True)
                _copy_or_symlink_tree(src, dst)
        else:
            staged_root = _find_staged_runtime_asset_root()
            if staged_root is None:
                raise FileNotFoundError(
                    "Unable to locate staged MyoSuite runtime assets or "
                    "vendored asset repositories."
                )
            asset_root.parent.mkdir(parents=True, exist_ok=True)
            _copy_or_symlink_tree(staged_root, asset_root)
        yield base_path


def _attach_default_configs(
    direct_entry_by_id: dict[str, dict[str, Any]], upstream_root: Path
) -> None:
    package_names = (
        "envpool",
        "envpool.mujoco",
        "envpool.mujoco.myosuite",
    )
    originals = {
        name: sys.modules.get(name)
        for name in (
            *package_names,
            "envpool.mujoco.myosuite.metadata",
            "envpool.mujoco.myosuite.paths",
            "_envpool_myosuite_config_codegen",
        )
    }
    for name in package_names:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = module

    metadata_module = types.ModuleType("envpool.mujoco.myosuite.metadata")
    metadata_module.MYOSUITE_DIRECT_ENTRIES = tuple(direct_entry_by_id.values())
    sys.modules["envpool.mujoco.myosuite.metadata"] = metadata_module

    paths_module = types.ModuleType("envpool.mujoco.myosuite.paths")
    paths_module.myosuite_asset_root = lambda: Path("__unused__")
    sys.modules["envpool.mujoco.myosuite.paths"] = paths_module

    config_path = _repo_root() / "envpool/mujoco/myosuite/config.py"
    spec = importlib.util.spec_from_file_location(
        "_envpool_myosuite_config_codegen",
        config_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load MyoSuite config module: {config_path}"
        )
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = config_module
    spec.loader.exec_module(config_module)
    generate_myosuite_task_config = config_module.generate_myosuite_task_config

    try:
        with _staged_runtime_asset_base(upstream_root) as base_path:
            for entry in direct_entry_by_id.values():
                entry["default_config"] = generate_myosuite_task_config(
                    entry,
                    {},
                    base_path=str(base_path),
                )
    finally:
        for name, module in originals.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def main() -> None:
    """Capture upstream suite registrations and emit canonical metadata."""
    args = _parse_args()
    upstream_root = args.upstream_root.resolve()
    myo_root = upstream_root / "myosuite" / "envs" / "myo"

    direct_entry_by_id: dict[str, dict[str, Any]] = {}
    for suite_name in ("myobase", "myochallenge", "myodm"):
        records, variant_records = _capture_suite_entries(
            upstream_root, suite_name
        )
        variant_map: dict[str, list[dict[str, Any]]] = {}
        for record in variant_records:
            variant_map.setdefault(record["env_id"], []).append({
                "variant_id": record["variant_id"],
                "variants": record["variants"],
            })
        for record in records:
            direct_entry_by_id[record["id"]] = _captured_entry_to_metadata(
                record,
                suite_name,
                suite_name,
                upstream_root,
                variant_map,
            )

    myoedits_records, myoedits_variant_records = _capture_suite_entries(
        upstream_root, "myoedits"
    )
    myoedits_variant_map: dict[str, list[dict[str, Any]]] = {}
    for record in myoedits_variant_records:
        myoedits_variant_map.setdefault(record["env_id"], []).append({
            "variant_id": record["variant_id"],
            "variants": record["variants"],
        })
    myoedits_direct = _direct_ids_from_register_env_with_variants(
        myo_root / "myoedits" / "__init__.py"
    )
    for record in myoedits_records:
        if record["id"] not in myoedits_direct:
            continue
        if record["id"] not in direct_entry_by_id:
            continue
        direct_entry_by_id[record["id"]] = _captured_entry_to_metadata(
            record,
            "myobase",
            "myoedits",
            upstream_root,
            myoedits_variant_map,
        )

    direct_entries = sorted(
        direct_entry_by_id.values(), key=lambda entry: entry["id"]
    )
    _attach_default_configs(direct_entry_by_id, upstream_root)
    direct_entries.sort(key=lambda entry: entry["id"])
    direct_ids = [entry["id"] for entry in direct_entries]
    expanded_ids = _dedupe_sorted(
        [entry["id"] for entry in direct_entries for _ in [None]]
        + [
            variant["variant_id"]
            for entry in direct_entries
            for variant in entry["variant_defs"]
        ]
    )

    myobase_direct = [
        entry["id"] for entry in direct_entries if entry["suite"] == "myobase"
    ]
    myochallenge_direct = [
        entry["id"]
        for entry in direct_entries
        if entry["suite"] == "myochallenge"
    ]
    myodm_direct = [
        entry["id"] for entry in direct_entries if entry["suite"] == "myodm"
    ]
    myodm_track_ids = [
        entry["id"]
        for entry in direct_entries
        if entry["suite"] == "myodm"
        and isinstance(entry["kwargs"].get("reference"), str)
    ]
    myodm_fixed_ids = [
        entry["id"]
        for entry in direct_entries
        if entry["id"].endswith("Fixed-v0")
    ]
    myodm_random_ids = [
        entry["id"]
        for entry in direct_entries
        if entry["id"].endswith("Random-v0")
    ]
    myodm_object_names = sorted({
        entry["kwargs"]["object_name"]
        for entry in direct_entries
        if entry["suite"] == "myodm"
    })
    data = {
        "pins": {
            "myosuite": {
                "version": args.version,
                "commit": args.commit,
            },
        },
        "suites": {
            "myobase_direct_ids": sorted(myobase_direct),
            "myochallenge_direct_ids": sorted(myochallenge_direct),
            "myodm_track_ids": sorted(myodm_track_ids),
            "myodm_object_names": myodm_object_names,
            "myodm_fixed_ids": sorted(
                entry_id
                for entry_id in myodm_fixed_ids
                if entry_id in myodm_direct
            ),
            "myodm_random_ids": sorted(
                entry_id
                for entry_id in myodm_random_ids
                if entry_id in myodm_direct
            ),
            "myoedits_direct_ids": sorted(myoedits_direct),
        },
        "counts": {
            "myobase_direct": len(myobase_direct),
            "myochallenge_direct": len(myochallenge_direct),
            "myodm_direct": len(myodm_direct),
            "direct_total": len(direct_ids),
            "expanded_total": len(expanded_ids),
        },
        "notes": {
            "myoedits_duplicate_ids": sorted(myoedits_direct),
            "variant_rules": [
                "Every myo* task also creates Sarc and Fati variants.",
                "Every myoHand* task also creates a Reaf variant.",
            ],
        },
        "direct_entries": direct_entries,
        "direct_ids": direct_ids,
        "expanded_ids": expanded_ids,
    }

    expected_expanded_ids = _dedupe_sorted(
        _expand_upstream_variants(direct_ids)
    )
    if expected_expanded_ids != expanded_ids:
        raise RuntimeError("Expanded IDs no longer match the expected rules")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
