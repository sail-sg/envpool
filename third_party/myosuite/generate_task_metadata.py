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

"""Generate compact native MyoSuite task metadata.

The input metadata is produced by:

  bazel run //envpool/mujoco:myosuite_oracle_probe -- \
      --mode metadata --out /tmp/myosuite_all_meta.json --task_id ...

Only the compact fields consumed by the native C++ runtime are emitted here.
The full official package remains a test/doc oracle, not a runtime dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCALAR_DEFAULTS = {
    "far_th": 0.0,
    "goal_th": 0.0,
    "hip_period": 0,
    "max_rot": 0.0,
    "min_height": 0.0,
    "pose_thd": 0.0,
    "target_x_vel": 0.0,
    "target_y_vel": 0.0,
}

ORACLE_BROKEN_SOURCE_METADATA = {
    "myosuite.envs.myo.myochallenge.bimanual_v0:BimanualEnvV1": {
        "obs_keys": [
            "time",
            "myohand_qpos",
            "myohand_qvel",
            "pros_hand_qpos",
            "pros_hand_qvel",
            "object_qpos",
            "object_qvel",
            "touching_body",
        ],
        "rwd_keys_wt": {
            "act": 0.0,
            "fin_dis": -0.5,
            "pass_err": -1.0,
            "reach_dist": -0.1,
        },
    },
    "myosuite.envs.myo.myochallenge.soccer_v0:SoccerEnvV0": {
        "obs_keys": [
            "internal_qpos",
            "internal_qvel",
            "grf",
            "torso_angle",
            "ball_pos",
            "model_root_pos",
            "model_root_vel",
            "muscle_length",
            "muscle_velocity",
            "muscle_force",
        ],
        "rwd_keys_wt": {
            "act_reg": -100.0,
            "goal_scored": 1000.0,
            "pain": -10.0,
            "time_cost": -0.01,
        },
    },
}


def _csv(items: list[Any] | None) -> str:
    if not items:
        return ""
    return ",".join(str(item) for item in items)


def _flat(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, list):
        out: list[Any] = []
        for value in values:
            if isinstance(value, list):
                out.extend(_flat(value))
            else:
                out.append(value)
        return out
    return [values]


def _float_csv(values: Any) -> str:
    return ",".join(f"{float(value):.17g}" for value in _flat(values))


def _rwd_csv(weights: dict[str, Any] | None) -> str:
    if not weights:
        return ""
    return ",".join(
        f"{key}:{float(weights[key]):.17g}" for key in sorted(weights)
    )


def _reach_range_csv(
    metadata: dict[str, Any], target_index: int
) -> tuple[str, str]:
    reach_range = metadata.get("target_reach_range")
    tip_sites = metadata.get("tip_sites") or []
    if not reach_range or target_index >= len(tip_sites):
        return "", ""
    span = reach_range.get(tip_sites[target_index])
    if span is None:
        return "", ""
    return _float_csv(span[0]), _float_csv(span[1])


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _entry(
    task: dict[str, Any], metadata: dict[str, Any] | None
) -> dict[str, Any]:
    if metadata is None:
        metadata = ORACLE_BROKEN_SOURCE_METADATA.get(task["entry_point"], {})
    low_ranges: list[str] = []
    high_ranges: list[str] = []
    for i, _target_site in enumerate(metadata.get("target_sites") or []):
        low, high = _reach_range_csv(metadata, i)
        low_ranges.append(low)
        high_ranges.append(high)
    entry = {
        "id": task["id"],
        "obs_keys": _csv(metadata.get("obs_keys")),
        "rwd_keys_wt": _rwd_csv(metadata.get("rwd_keys_wt")),
        "init_qpos": _float_csv(metadata.get("init_qpos")),
        "init_qvel": _float_csv(metadata.get("init_qvel")),
        "reset_qacc_warmstart": _float_csv(
            (metadata.get("reset_state") or {}).get("qacc_warmstart")
        ),
        "target_jnt_value": _float_csv(metadata.get("target_jnt_value")),
        "tip_sites": _csv(metadata.get("tip_sites")),
        "target_sites": _csv(metadata.get("target_sites")),
        "target_reach_low": ";".join(low_ranges),
        "target_reach_high": ";".join(high_ranges),
        "reset_type": str(metadata.get("reset_type", "")),
    }
    for key, default in SCALAR_DEFAULTS.items():
        value = metadata.get(key, default)
        entry[key] = default if value is None else value
    return entry


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
        "// Generated from pinned MyoSuite oracle metadata; do not edit by hand.",
        "#ifndef THIRD_PARTY_MYOSUITE_MYOSUITE_TASK_METADATA_H_",
        "#define THIRD_PARTY_MYOSUITE_MYOSUITE_TASK_METADATA_H_",
        "",
        "#include <array>",
        "#include <stdexcept>",
        "#include <string_view>",
        "",
        "namespace third_party::myosuite {",
        "",
        "struct MyoSuiteTaskMetadata {",
        "  const char* id;",
        "  const char* obs_keys;",
        "  const char* rwd_keys_wt;",
        "  const char* init_qpos;",
        "  const char* init_qvel;",
        "  const char* reset_qacc_warmstart;",
        "  const char* target_jnt_value;",
        "  const char* tip_sites;",
        "  const char* target_sites;",
        "  const char* target_reach_low;",
        "  const char* target_reach_high;",
        "  const char* reset_type;",
        "  double far_th;",
        "  double goal_th;",
        "  int hip_period;",
        "  double max_rot;",
        "  double min_height;",
        "  double pose_thd;",
        "  double target_x_vel;",
        "  double target_y_vel;",
        "};",
        "",
        "// clang-format off",
        (
            f"inline constexpr std::array<MyoSuiteTaskMetadata, {len(entries)}> "
            "kMyoSuiteTaskMetadata = {{"
        ),
    ]
    for entry in entries:
        lines.extend([
            "    MyoSuiteTaskMetadata{",
            f'        "{_escape(entry["id"])}",',
            f'        "{_escape(entry["obs_keys"])}",',
            f'        "{_escape(entry["rwd_keys_wt"])}",',
            f'        "{_escape(entry["init_qpos"])}",',
            f'        "{_escape(entry["init_qvel"])}",',
            f'        "{_escape(entry["reset_qacc_warmstart"])}",',
            f'        "{_escape(entry["target_jnt_value"])}",',
            f'        "{_escape(entry["tip_sites"])}",',
            f'        "{_escape(entry["target_sites"])}",',
            f'        "{_escape(entry["target_reach_low"])}",',
            f'        "{_escape(entry["target_reach_high"])}",',
            f'        "{_escape(entry["reset_type"])}",',
            f"        {float(entry['far_th']):.17g},",
            f"        {float(entry['goal_th']):.17g},",
            f"        {int(entry['hip_period'])},",
            f"        {float(entry['max_rot']):.17g},",
            f"        {float(entry['min_height']):.17g},",
            f"        {float(entry['pose_thd']):.17g},",
            f"        {float(entry['target_x_vel']):.17g},",
            f"        {float(entry['target_y_vel']):.17g},",
            "    },",
        ])
    lines.extend([
        "}};",
        "// clang-format on",
        "",
        "inline const MyoSuiteTaskMetadata& GetMyoSuiteTaskMetadata(",
        "    std::string_view task_id) {",
        "  for (const auto& metadata : kMyoSuiteTaskMetadata) {",
        "    if (metadata.id == task_id) {",
        "      return metadata;",
        "    }",
        "  }",
        '  throw std::runtime_error("Unknown MyoSuite task metadata.");',
        "}",
        "",
        "}  // namespace third_party::myosuite",
        "",
        "#endif  // THIRD_PARTY_MYOSUITE_MYOSUITE_TASK_METADATA_H_",
        "",
    ])
    output.write_text("\n".join(lines))


def main() -> None:
    """Generate compact C++ and JSON task metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--oracle-metadata", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-header", type=Path, required=True)
    args = parser.parse_args()

    tasks = json.loads(args.tasks.read_text())
    oracle = json.loads(args.oracle_metadata.read_text())["tasks"]
    entries = [_entry(task, oracle.get(task["id"])) for task in tasks]
    args.out_json.write_text(
        json.dumps(entries, indent=2, sort_keys=True) + "\n"
    )
    _write_header(entries, args.out_header)


if __name__ == "__main__":
    main()
