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

"""Generate native MyoDM reference trajectories from pinned MyoSuite assets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def _flat(values: Any) -> list[float]:
    if values is None:
        return []
    array = np.asarray(values, dtype=np.float64)
    return [float(value) for value in array.ravel()]


def _shape(values: Any) -> tuple[int, int]:
    if values is None:
        return (0, 0)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return (1, int(array.shape[0]))
    if array.ndim == 2:
        return (int(array.shape[0]), int(array.shape[1]))
    raise ValueError(f"expected 1D or 2D reference array, got {array.shape}")


def _fixed_reference(randomized: bool) -> dict[str, Any]:
    dof_robot = 29
    if randomized:
        return {
            "type": "kRandom",
            "time": [0.0, 4.0],
            "robot": np.zeros((2, dof_robot)),
            "robot_vel": np.zeros((2, dof_robot)),
            "object": np.asarray([
                [-0.2, -0.2, 0.1, 1.0, 0.0, 0.0, -1.0],
                [0.2, 0.2, 0.1, 1.0, 0.0, 0.0, 1.0],
            ]),
            "robot_init": np.zeros(dof_robot),
            "object_init": np.asarray([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]),
        }
    return {
        "type": "kFixed",
        "time": [0.0, 4.0],
        "robot": np.zeros((1, dof_robot)),
        "robot_vel": np.zeros((1, dof_robot)),
        "object": np.asarray([[0.2, 0.2, 0.1, 1.0, 0.0, 0.0, 0.1]]),
        "robot_init": np.zeros(dof_robot),
        "object_init": np.asarray([-0.2, -0.2, 0.1, 1.0, 0.0, 0.0, 0.0]),
    }


def _reference_entry(task: dict[str, Any], source_root: Path) -> dict[str, Any]:
    ref_path = task.get("reference_path") or ""
    if ref_path:
        path = source_root / ref_path
        data = dict(np.load(path).items())
        data.setdefault("robot_vel", None)
        data["type"] = "kTrack"
    else:
        data = _fixed_reference(task["id"].endswith("Random-v0"))

    robot_rows, robot_cols = _shape(data.get("robot"))
    robot_vel_rows, robot_vel_cols = _shape(data.get("robot_vel"))
    object_rows, object_cols = _shape(data.get("object"))
    return {
        "id": task["id"],
        "type": data["type"],
        "time": _flat(data.get("time")),
        "robot": _flat(data.get("robot")),
        "robot_rows": robot_rows,
        "robot_cols": robot_cols,
        "robot_vel": _flat(data.get("robot_vel")),
        "robot_vel_rows": robot_vel_rows,
        "robot_vel_cols": robot_vel_cols,
        "object": _flat(data.get("object")),
        "object_rows": object_rows,
        "object_cols": object_cols,
        "robot_init": _flat(data.get("robot_init")),
        "object_init": _flat(data.get("object_init")),
    }


def _name(task_id: str, suffix: str) -> str:
    stem = re.sub(r"[^0-9A-Za-z]", "_", task_id)
    return f"kMyoSuiteReference_{stem}_{suffix}"


def _array(name: str, values: list[float]) -> list[str]:
    lines = [f"inline constexpr std::array<double, {len(values)}> {name} = {{"]
    if values:
        for start in range(0, len(values), 6):
            chunk = values[start : start + 6]
            lines.append(
                "    " + ", ".join(f"{value:.17g}" for value in chunk) + ","
            )
    lines.append("};")
    return lines


def _write_header(entries: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "// Copyright 2026 Garena Online Private Limited",
        "//",
        '// Licensed under the Apache License, Version 2.0 (the "License");',
        "// you may not use this file except in compliance with the License.",
        "// You may obtain a copy of the License at",
        "//",
        "//      http://www.apache.org/licenses/LICENSE-2.0",
        "//",
        "// Unless required by applicable law or agreed to in writing, software",
        '// distributed under the License is distributed on an "AS IS" BASIS,',
        "// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
        "// See the License for the specific language governing permissions and",
        "// limitations under the License.",
        "",
        "// Generated from pinned MyoSuite MyoDM reference assets; do not edit by hand.",
        "#ifndef THIRD_PARTY_MYOSUITE_MYOSUITE_REFERENCE_DATA_H_",
        "#define THIRD_PARTY_MYOSUITE_MYOSUITE_REFERENCE_DATA_H_",
        "",
        "#include <array>",
        "#include <cstdint>",
        "#include <stdexcept>",
        "#include <string>",
        "#include <string_view>",
        "",
        "namespace third_party::myosuite {",
        "",
        "enum class MyoSuiteReferenceType : std::uint8_t {",
        "  kNone,",
        "  kFixed,",
        "  kRandom,",
        "  kTrack,",
        "};",
        "",
        "struct MyoSuiteReferenceData {",
        "  const char* id;",
        "  MyoSuiteReferenceType type;",
        "  const double* time;",
        "  int time_size;",
        "  const double* robot;",
        "  int robot_rows;",
        "  int robot_cols;",
        "  const double* robot_vel;",
        "  int robot_vel_rows;",
        "  int robot_vel_cols;",
        "  const double* object;",
        "  int object_rows;",
        "  int object_cols;",
        "  const double* robot_init;",
        "  int robot_init_size;",
        "  const double* object_init;",
        "  int object_init_size;",
        "};",
        "",
        "inline constexpr std::array<double, 0> kMyoSuiteEmptyReference = {};",
        "",
        "// clang-format off",
    ]
    for entry in entries:
        for key in (
            "time",
            "robot",
            "robot_vel",
            "object",
            "robot_init",
            "object_init",
        ):
            lines.extend(_array(_name(entry["id"], key), entry[key]))
            lines.append("")

    lines.append(
        f"inline constexpr std::array<MyoSuiteReferenceData, {len(entries)}> "
        "kMyoSuiteReferenceData = {{"
    )
    for entry in entries:
        ident = entry["id"]
        lines.extend([
            "    MyoSuiteReferenceData{",
            f'        "{ident}",',
            f"        MyoSuiteReferenceType::{entry['type']},",
            f"        {_name(ident, 'time')}.data(),",
            f"        {len(entry['time'])},",
            f"        {_name(ident, 'robot')}.data(),",
            f"        {entry['robot_rows']},",
            f"        {entry['robot_cols']},",
            f"        {_name(ident, 'robot_vel')}.data(),",
            f"        {entry['robot_vel_rows']},",
            f"        {entry['robot_vel_cols']},",
            f"        {_name(ident, 'object')}.data(),",
            f"        {entry['object_rows']},",
            f"        {entry['object_cols']},",
            f"        {_name(ident, 'robot_init')}.data(),",
            f"        {len(entry['robot_init'])},",
            f"        {_name(ident, 'object_init')}.data(),",
            f"        {len(entry['object_init'])},",
            "    },",
        ])
    lines.extend([
        "}};",
        "// clang-format on",
        "",
        "inline constexpr MyoSuiteReferenceData kEmptyMyoSuiteReferenceData = {",
        '    "",',
        "    MyoSuiteReferenceType::kNone,",
        "    kMyoSuiteEmptyReference.data(),",
        "    0,",
        "    kMyoSuiteEmptyReference.data(),",
        "    0,",
        "    0,",
        "    kMyoSuiteEmptyReference.data(),",
        "    0,",
        "    0,",
        "    kMyoSuiteEmptyReference.data(),",
        "    0,",
        "    0,",
        "    kMyoSuiteEmptyReference.data(),",
        "    0,",
        "    kMyoSuiteEmptyReference.data(),",
        "    0,",
        "};",
        "",
        "inline const MyoSuiteReferenceData& GetMyoSuiteReferenceData(",
        "    std::string_view task_id) {",
        "  for (const auto& reference : kMyoSuiteReferenceData) {",
        "    if (reference.id == task_id) {",
        "      return reference;",
        "    }",
        "  }",
        "  return kEmptyMyoSuiteReferenceData;",
        "}",
        "",
        "}  // namespace third_party::myosuite",
        "",
        "#endif  // THIRD_PARTY_MYOSUITE_MYOSUITE_REFERENCE_DATA_H_",
        "",
    ])
    output.write_text("\n".join(lines))


def main() -> None:
    """Generate the C++ MyoDM reference data header."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--myosuite-source", type=Path, required=True)
    parser.add_argument("--out-header", type=Path, required=True)
    args = parser.parse_args()

    tasks = json.loads(args.tasks.read_text())
    entries = [
        _reference_entry(task, args.myosuite_source)
        for task in tasks
        if task["kind"] == "kMyoDmTrack"
    ]
    _write_header(entries, args.out_header)


if __name__ == "__main__":
    main()
