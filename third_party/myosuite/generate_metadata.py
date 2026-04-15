#!/usr/bin/env python3
"""Generate canonical MyoSuite registry metadata from upstream source."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable


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
        if isinstance(node, ast.Call) and _call_name(node) == "register_env_with_variants":
            env_id = _keyword_string(node, "id")
            if env_id is not None:
                ids.append(env_id)
    return ids


def _string_tuple_items(path: Path, target_name: str) -> list[str]:
    module = _load_module(path)
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == target_name:
                    if isinstance(node.value, ast.Tuple):
                        values: list[str] = []
                        for element in node.value.elts:
                            value = _constant_string(element)
                            if value is not None:
                                values.append(value)
                        return values
    raise ValueError(f"Could not find tuple assignment for {target_name} in {path}")


def _myodm_track_ids(path: Path) -> list[str]:
    module = _load_module(path)
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MyoHand_task_spec":
                    if not isinstance(node.value, ast.Tuple):
                        raise ValueError("MyoHand_task_spec is not a tuple")
                    ids: list[str] = []
                    for entry in node.value.elts:
                        if not isinstance(entry, ast.Call):
                            continue
                        name = _keyword_string(entry, "name")
                        if name is not None:
                            ids.append(name)
                    return ids
    raise ValueError(f"Could not find MyoHand_task_spec in {path}")


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    upstream_root = args.upstream_root.resolve()
    myosuite_root = upstream_root / "myosuite"
    myo_root = myosuite_root / "envs" / "myo"

    myobase_direct = _direct_ids_from_register_env_with_variants(
        myo_root / "myobase" / "__init__.py"
    )
    myochallenge_direct = _direct_ids_from_register_env_with_variants(
        myo_root / "myochallenge" / "__init__.py"
    )
    myoedits_direct = _direct_ids_from_register_env_with_variants(
        myo_root / "myoedits" / "__init__.py"
    )

    myodm_track_ids = _myodm_track_ids(myo_root / "myodm" / "__init__.py")
    myodm_objects = _string_tuple_items(myo_root / "myodm" / "__init__.py", "OBJECTS")
    myodm_fixed_ids = [f"MyoHand{obj.title()}Fixed-v0" for obj in myodm_objects]
    myodm_random_ids = [f"MyoHand{obj.title()}Random-v0" for obj in myodm_objects]
    myodm_direct = myodm_track_ids + myodm_fixed_ids + myodm_random_ids

    direct_ids = _dedupe_sorted(
        myobase_direct + myochallenge_direct + myodm_direct
    )
    expanded_ids = _dedupe_sorted(_expand_upstream_variants(direct_ids))

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
            "myodm_object_names": sorted(myodm_objects),
            "myodm_fixed_ids": sorted(myodm_fixed_ids),
            "myodm_random_ids": sorted(myodm_random_ids),
            "myoedits_direct_ids": sorted(myoedits_direct),
        },
        "counts": {
            "myobase_direct": len(set(myobase_direct)),
            "myochallenge_direct": len(set(myochallenge_direct)),
            "myodm_direct": len(set(myodm_direct)),
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
        "direct_ids": direct_ids,
        "expanded_ids": expanded_ids,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
