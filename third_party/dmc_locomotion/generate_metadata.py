# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract locomotion entry points and model parameters without importing DMC."""

import argparse
import ast
import json
from pathlib import Path


def assignments(path: Path) -> dict[str, ast.expr]:
    """Read top-level named assignments without executing upstream code."""
    return {
        target.id: node.value
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def generate(source: Path, labmaze: Path, header: Path, registry: Path) -> None:
    """Generate registry and model metadata from the pinned official source."""
    if source.is_file():
        source = source.parent
    tasks = []
    for path in sorted((source / "locomotion/examples").glob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if [arg.arg for arg in node.args.args] != ["random_state"]:
                continue
            tasks.append(node.name)
    soccer = ast.parse((source / "locomotion/soccer/__init__.py").read_text())
    walkers = [
        target.id.lower()
        for node in soccer.body
        if isinstance(node, ast.ClassDef) and node.name == "WalkerType"
        for member in node.body
        if isinstance(member, ast.Assign)
        for target in member.targets
        if isinstance(target, ast.Name)
    ]
    tasks.extend(f"soccer_{walker}" for walker in walkers)
    registry.write_text(json.dumps(tasks, indent=2) + "\n")
    cmu = assignments(source / "locomotion/walkers/cmu_humanoid.py")
    pitch = assignments(source / "locomotion/soccer/pitch.py")
    textures = assignments(labmaze)
    blocks = [
        "// Generated from the pinned dm_control source. Do not edit.",
        "#ifndef THIRD_PARTY_DMC_LOCOMOTION_METADATA_H_",
        "#define THIRD_PARTY_DMC_LOCOMOTION_METADATA_H_",
        "#include <array>",
        "#include <string_view>",
        "namespace mujoco_locomotion {",
        "struct PositionActuator {",
        "  const char* name; double low, high, kp, damping;",
        "};",
    ]
    for variable, name in (
        ("_POSITION_ACTUATORS", "kCmu2019Actuators"),
        ("_POSITION_ACTUATORS_V2020", "kCmu2020Actuators"),
    ):
        values = cmu[variable]
        assert isinstance(values, ast.List)
        blocks.append(
            f"inline constexpr std::array<PositionActuator, {len(values.elts)}> "
            f"{name} = {{{{"
        )
        for value in values.elts:
            assert isinstance(value, ast.Call)
            args = [ast.literal_eval(arg) for arg in value.args]
            joint, limits, kp = args[:3]
            damping = args[3] if len(args) == 4 else -1.0
            blocks.append(
                f"  {{{json.dumps(joint)}, {float(limits[0])}, "
                f"{float(limits[1])}, {float(kp)}, {float(damping)}}},"
            )
        blocks.append("}};")
    for name, values in (
        ("kCmuMocapJoints", ast.literal_eval(cmu["_CMU_MOCAP_JOINTS"])),
        ("kTaskNames", tasks),
        (
            "kWallTextures",
            ast.literal_eval(textures["WALL_TEXTURES"])["style_01"],
        ),
        (
            "kFloorTextures",
            ast.literal_eval(textures["FLOOR_TEXTURES"])["style_01"],
        ),
    ):
        blocks.append(
            f"inline constexpr std::array<std::string_view, {len(values)}> "
            f"{name} = {{{{"
        )
        blocks.extend(f"  {json.dumps(value)}," for value in values)
        blocks.append("}};")
    posts = ast.literal_eval(pitch["_GOALPOSTS"])
    blocks.extend([
        "struct GoalPost { std::string_view name; std::array<double, 6> fromto; };",
        f"inline constexpr std::array<GoalPost, {len(posts)}> kGoalPosts = {{{{",
    ])
    for name, coordinates in posts.items():
        values = ", ".join(str(float(x)) for x in coordinates)
        blocks.append(f"  {{{json.dumps(name)}, {{{values}}}}},")
    blocks.append("}};")
    blocks.extend([
        "}  // namespace mujoco_locomotion",
        "#endif  // THIRD_PARTY_DMC_LOCOMOTION_METADATA_H_",
    ])
    header.write_text("\n".join(blocks) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--labmaze", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.labmaze, args.header, args.registry)
