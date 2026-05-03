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

"""Regenerate native MyoSuite task registry files.

The source registry is the pinned MyoSuite task list checked in as JSON.  The
optional oracle metadata input is produced by `myosuite_oracle_probe --mode
metadata` and refreshes space fields that can change with the pinned MuJoCo
runtime and upstream MjSpec patching.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ORACLE_VERSION = "2.11.6"
ORACLE_COMMIT = "05cb84678373f91271004f99602ebbf01e57d1a1"

BROKEN_IDS = (
    "myoChallengeBimanual-v0",
    "myoChallengeSoccerP1-v0",
    "myoChallengeSoccerP2-v0",
    "myoFatiChallengeBimanual-v0",
    "myoFatiChallengeSoccerP1-v0",
    "myoFatiChallengeSoccerP2-v0",
    "myoSarcChallengeBimanual-v0",
    "myoSarcChallengeSoccerP1-v0",
    "myoSarcChallengeSoccerP2-v0",
)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _refresh_from_metadata(
    tasks: list[dict[str, Any]], metadata_path: Path | None
) -> None:
    if metadata_path is None:
        return
    oracle = json.loads(metadata_path.read_text())
    if oracle["version"] != ORACLE_VERSION:
        raise ValueError(
            f"expected MyoSuite {ORACLE_VERSION}, got {oracle['version']}"
        )
    by_id = oracle["tasks"]
    for task in tasks:
        metadata = by_id.get(task["id"])
        if metadata is None:
            continue
        task["obs_dim"] = int(metadata["observation_shape"][0])
        task["action_dim"] = int(metadata["action_shape"][0])
        task["frame_skip"] = int(metadata["frame_skip"])
        task["oracle_numpy2_broken"] = False
    for task in tasks:
        if task["id"] in BROKEN_IDS:
            task["oracle_numpy2_broken"] = True


def _write_json(tasks: list[dict[str, Any]], output: Path) -> None:
    output.write_text(json.dumps(tasks, indent=2, sort_keys=True) + "\n")


def _write_header(tasks: list[dict[str, Any]], output: Path) -> None:
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
        f"// Generated from MyoSuite v{ORACLE_VERSION} registry; do not edit by hand.",
        "#ifndef THIRD_PARTY_MYOSUITE_MYOSUITE_TASKS_H_",
        "#define THIRD_PARTY_MYOSUITE_MYOSUITE_TASKS_H_",
        "",
        "#include <array>",
        "#include <cstdint>",
        "#include <stdexcept>",
        "#include <string>",
        "#include <string_view>",
        "",
        "namespace third_party::myosuite {",
        "",
        "enum class MyoSuiteTaskKind : std::uint8_t {",
        "  kPose,",
        "  kReach,",
        "  kWalkReach,",
        "  kWalk,",
        "  kTerrain,",
        "  kKeyTurn,",
        "  kObjHoldFixed,",
        "  kObjHoldRandom,",
        "  kPenTwirlFixed,",
        "  kPenTwirlRandom,",
        "  kTorsoPose,",
        "  kReorientSar,",
        "  kChallengeBaoding,",
        "  kChallengeBimanual,",
        "  kChallengeChaseTag,",
        "  kChallengeRelocate,",
        "  kChallengeReorient,",
        "  kChallengeRunTrack,",
        "  kChallengeSoccer,",
        "  kChallengeTableTennis,",
        "  kMyoDmTrack,",
        "};",
        "",
        "enum class MyoSuiteMuscleCondition : std::uint8_t {",
        "  kNormal,",
        "  kSarcopenia,",
        "  kFatigue,",
        "  kReafferentation,",
        "};",
        "",
        "struct MyoSuiteTaskDef {",
        "  const char* id;",
        "  const char* envpool_id;",
        "  const char* entry_point;",
        "  MyoSuiteTaskKind kind;",
        "  const char* model_path;",
        "  const char* reference_path;",
        "  const char* object_name;",
        "  int obs_dim;",
        "  int action_dim;",
        "  int max_episode_steps;",
        "  int frame_skip;",
        "  bool normalize_act;",
        "  MyoSuiteMuscleCondition muscle_condition;",
        "  bool oracle_numpy2_broken;",
        "};",
        "",
        "// clang-format off",
        (
            f"inline constexpr std::array<MyoSuiteTaskDef, {len(tasks)}> "
            "kMyoSuiteTasks = {{"
        ),
    ]
    for task in tasks:
        lines.extend([
            "    MyoSuiteTaskDef{",
            f'        "{_escape(task["id"])}",',
            f'        "MyoSuite/{_escape(task["id"])}",',
            f'        "{_escape(task["entry_point"])}",',
            f"        MyoSuiteTaskKind::{task['kind']},",
            f'        "{_escape(task["model_path"])}",',
            f'        "{_escape(task["reference_path"])}",',
            f'        "{_escape(task["object_name"])}",',
            f"        {int(task['obs_dim'])},",
            f"        {int(task['action_dim'])},",
            f"        {int(task['max_episode_steps'])},",
            f"        {int(task['frame_skip'])},",
            f"        {_bool(bool(task['normalize_act']))},",
            f"        MyoSuiteMuscleCondition::{task['muscle']},",
            f"        {_bool(bool(task['oracle_numpy2_broken']))},",
            "    },",
        ])
    lines.extend([
        "}};",
        "// clang-format on",
        "",
        "inline const MyoSuiteTaskDef& GetMyoSuiteTask(std::string_view id) {",
        "  for (const auto& task : kMyoSuiteTasks) {",
        "    if (task.id == id || task.envpool_id == id) {",
        "      return task;",
        "    }",
        "  }",
        '  throw std::runtime_error("Unknown MyoSuite task: " + std::string(id));',
        "}",
        "",
        "}  // namespace third_party::myosuite",
        "",
        "#endif  // THIRD_PARTY_MYOSUITE_MYOSUITE_TASKS_H_",
        "",
    ])
    output.write_text("\n".join(lines))


def main() -> None:
    """Generate native task registry files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-json", type=Path, required=True)
    parser.add_argument("--oracle-metadata", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-header", type=Path, required=True)
    args = parser.parse_args()

    tasks = json.loads(args.tasks_json.read_text())
    _refresh_from_metadata(tasks, args.oracle_metadata)
    _write_json(tasks, args.out_json)
    _write_header(tasks, args.out_header)


if __name__ == "__main__":
    main()
