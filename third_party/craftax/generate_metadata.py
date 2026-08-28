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
"""Extract native enum metadata without importing or executing upstream code."""

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def identifier(node: ast.AST) -> str:
    """Require an unqualified name in declarative upstream metadata."""
    if not isinstance(node, ast.Name):
        raise ValueError(f"Expected a name, got {ast.dump(node)}")
    return node.id


def texture_metadata(tree: ast.Module) -> list[str]:
    """Keep native texture ordering tied to the official renderer's tables."""
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "load_all_textures_given_size"
    )
    output = []
    wanted = {
        "texture_names": "BLOCK_TEXTURE_NAMES",
        "block_texture_names": "BLOCK_TEXTURE_NAMES",
        "item_texture_names": "ITEM_TEXTURE_NAMES",
        "player_textures": "PLAYER_TEXTURE_NAMES",
        "melee_mob_textures": "MELEE_TEXTURE_NAMES",
        "passive_mob_textures": "PASSIVE_TEXTURE_NAMES",
        "ranged_mob_textures": "RANGED_TEXTURE_NAMES",
        "projectile_textures": "PROJECTILE_TEXTURE_NAMES",
    }
    for node in function.body:
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        name = (
            identifier(target.elts[0])
            if isinstance(target, ast.Tuple)
            else getattr(target, "id", "")
        )
        if name not in wanted:
            continue
        filenames = [
            part.value
            for part in ast.walk(node.value)
            if isinstance(part, ast.Constant)
            and isinstance(part.value, str)
            and part.value.endswith(".png")
        ]
        if filenames:
            output.append(
                f"inline constexpr const char* {wanted[name]}[] = {{"
                + ", ".join(map(json.dumps, filenames))
                + "};"
            )
    return output


def literal(node: ast.AST, bindings: dict[str, Any]) -> Any:
    """Read the literal arrays used by upstream game metadata."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [literal(item, bindings) for item in node.elts]
    if isinstance(node, ast.Name):
        return bindings[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -literal(node.operand, bindings)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return literal(node.left, bindings) * literal(node.right, bindings)
    if isinstance(node, ast.Attribute):
        value = literal(node.value, bindings)
        return value if node.attr == "value" else value[node.attr]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "array"
    ):
        return literal(node.args[0], bindings)
    raise ValueError(ast.dump(node))


def array_declaration(name: str, value: Any) -> str:
    """Emit a rectangular C++ array."""
    shape = []
    item = value
    while isinstance(item, list):
        shape.append(len(item))
        item = item[0]

    def leaves(item: Any) -> list[Any]:
        if isinstance(item, list):
            return [leaf for child in item for leaf in leaves(child)]
        return [item]

    floating = any(isinstance(item, float) for item in leaves(value))

    def contents(item: Any) -> str:
        if isinstance(item, list):
            return "{" + ", ".join(contents(child) for child in item) + "}"
        return repr(float(item)) + "f" if floating else str(int(item))

    dims = "".join(f"[{dim}]" for dim in shape)
    return f"inline constexpr {'float' if floating else 'int'} {name}{dims} = {contents(value)};"


def world_metadata(path: Path, blocks: dict) -> list[str]:
    """Read the upstream world configuration dataclasses and their instances."""
    tree = ast.parse(path.read_text())
    fields = {
        node.name: [
            identifier(entry.target)
            for entry in node.body
            if isinstance(entry, ast.AnnAssign)
        ]
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    instances = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(
            node.value, ast.Call
        ):
            continue
        call = node.value
        if isinstance(call.func, ast.Name) and call.func.id in fields:
            values = {
                arg.arg: literal(arg.value, {"BlockType": blocks})
                for arg in call.keywords
            }
            instances[identifier(node.targets[0])] = (call.func.id, values)
    output = ["namespace full {"]
    for cls, names in fields.items():
        output.append(f"struct {cls} {{")
        for name in names:
            values = [v[name] for c, v in instances.values() if c == cls]
            leaf = [
                v
                for value in values
                for v in (value if isinstance(value, list) else [value])
            ]
            dtype = (
                "float" if any(isinstance(v, float) for v in leaf) else "int"
            )
            field_type = (
                f"std::array<{dtype}, {len(values[0])}>"
                if isinstance(values[0], list)
                else dtype
            )
            output.append(f"  {field_type} {name};")
        output.append("};")

    def contents(value: Any) -> str:
        if isinstance(value, list):
            return "{" + ", ".join(contents(v) for v in value) + "}"
        return (
            repr(value) + "f" if isinstance(value, float) else str(int(value))
        )

    for name, (cls, values) in instances.items():
        output.append(
            f"inline constexpr {cls} {name} = {{"
            + ", ".join(contents(values[field]) for field in fields[cls])
            + "};"
        )
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and identifier(node.targets[0]).startswith("ALL_")
        ):
            assert isinstance(node.value, ast.Call)
            names = [identifier(arg) for arg in node.value.args[1:]]
            cls = instances[names[0]][0]
            output.append(
                f"inline constexpr {cls} {identifier(node.targets[0])}[] = {{"
                + ", ".join(names)
                + "};"
            )
    output.append("}  // namespace full")
    return output


def generate(classic: Path, full: Path, world: Path | None = None) -> str:
    """Generate enums directly from the pinned source declarations."""
    out = [
        "// Generated from Craftax v1.6.1; do not edit.",
        "// Upstream copyright (c) 2024 Michael Matthews, MIT license.",
        "#ifndef THIRD_PARTY_CRAFTAX_CONSTANTS_H_",
        "#define THIRD_PARTY_CRAFTAX_CONSTANTS_H_",
        "#include <array>",
        "namespace craftax {",
    ]
    for family, path in (("classic", classic), ("full", full)):
        out.append(f"namespace {family} {{")
        tree = ast.parse(path.read_text())
        out.extend(texture_metadata(tree))
        bindings = {}
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            namespace = {
                "BlockType": "block",
                "ItemType": "item",
                "Action": "action",
                "Achievement": "achievement",
                "MobType": "mob",
                "ProjectileType": "projectile",
            }.get(node.name)
            if namespace is None:
                continue
            out.extend([f"namespace {namespace} {{", "enum Value : int {"])
            count = 0
            enum = {}
            for entry in node.body:
                if isinstance(entry, ast.Assign):
                    name = entry.targets[0]
                    assert isinstance(name, ast.Name)
                    value = ast.literal_eval(entry.value)
                    assert isinstance(value, int)
                    out.append(f"  {name.id} = {value},")
                    enum[name.id] = value
                    count += 1
            out.extend([
                f"  COUNT = {count}",
                "};",
                f"}}  // namespace {namespace}",
            ])
            bindings[node.name] = enum
        wanted = {
            "FLOOR_MOB_MAPPING",
            "FLOOR_MOB_SPAWN_CHANCE",
            "MOB_TYPE_COLLISION_MAPPING",
            "MOB_TYPE_DAMAGE_MAPPING",
            "MOB_TYPE_HEALTH_MAPPING",
            "MOB_TYPE_DEFENSE_MAPPING",
            "RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING",
            "SOLID_BLOCKS",
            "CAN_PLACE_ITEM_BLOCKS",
            "LEVEL_ACHIEVEMENT_MAP",
            "MOB_ACHIEVEMENT_MAP",
            "INTERMEDIATE_ACHIEVEMENTS",
            "VERY_ADVANCED_ACHIEVEMENTS",
            "MONSTERS_KILLED_TO_CLEAR_LEVEL",
            "BOSS_FIGHT_EXTRA_DAMAGE",
            "BOSS_FIGHT_SPAWN_TURNS",
        }
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(
                node.targets[0], ast.Name
            ):
                continue
            name = identifier(node.targets[0])
            try:
                value = literal(node.value, bindings)
            except (ValueError, KeyError, TypeError):
                if name in wanted:
                    raise
                continue
            bindings[name] = value
            if name in wanted:
                out.append(array_declaration(name, value))
        out.append(f"}}  // namespace {family}")
    if world is not None:
        out.extend(world_metadata(world, bindings["BlockType"]))
    out.extend([
        "}  // namespace craftax",
        "#endif  // THIRD_PARTY_CRAFTAX_CONSTANTS_H_",
        "",
    ])
    return "\n".join(out)


def info_metadata(classic: Path, full: Path) -> str:
    """Generate typed EnvPool information keys from upstream achievements."""
    lines = [
        "// Generated from Craftax v1.6.1; MIT license.",
        "#ifndef THIRD_PARTY_CRAFTAX_INFO_H_",
        "#define THIRD_PARTY_CRAFTAX_INFO_H_",
        "namespace craftax {",
        "template <bool Classic> struct AchievementInfo;",
    ]
    for is_classic, path in ((True, classic), (False, full)):
        tree = ast.parse(path.read_text())
        enum = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Achievement"
        )
        fields = [
            (identifier(node.targets[0]).lower(), ast.literal_eval(node.value))
            for node in enum.body
            if isinstance(node, ast.Assign)
        ]
        lines.append(
            "template <> struct AchievementInfo<"
            + str(is_classic).lower()
            + "> {"
        )
        lines.append("static auto StateSpec() { return MakeDict(")
        lines.append(
            ",\n".join(
                f'"info:Achievements/{name}"_.Bind(Spec<float>({{}}, {{0, 100}}))'
                for name, _ in fields
            )
            + "); }"
        )
        lines.append(
            "template <typename State> static void Write(State& out, const std::vector<std::uint8_t>& achievements, bool done) {"
        )
        lines.extend(
            f'out["info:Achievements/{name}"_] = static_cast<float>(achievements[{index}] * done * 100);'
            for name, index in fields
        )
        lines.append("} }; ")
    lines.extend([
        "}  // namespace craftax",
        "#endif  // THIRD_PARTY_CRAFTAX_INFO_H_",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    """Write the generated header."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classic", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--world", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--info-output", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--registry-output", type=Path)
    args = parser.parse_args()
    args.output.write_text(generate(args.classic, args.full, args.world))
    if args.info_output:
        args.info_output.write_text(info_metadata(args.classic, args.full))
    if args.registry_output:
        tree = ast.parse(args.registry.read_text())
        names = sorted({
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("Craftax-")
            and node.value.endswith("-v1")
        })
        license_header = Path(__file__).read_text().split('"""', 1)[0]
        args.registry_output.write_text(
            license_header
            + '"""Generated from the pinned upstream factory; do not edit."""\n\n'
            + "CRAFTAX_IDS = (\n"
            + "".join(f'    "{name}",\n' for name in names)
            + ")\n"
        )


if __name__ == "__main__":
    main()
