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
"""Generate the native task registry and defaults from pinned MPE2 source."""

import argparse
import ast
import re
from pathlib import Path


def main() -> None:
    """Emit C++ task metadata and the licensed upstream font."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source.parent
    fields = (
        "N",
        "num_good",
        "num_adversaries",
        "num_obstacles",
        "num_food",
        "num_forests",
        "num_collectors",
        "num_deposits",
        "num_treasures",
    )
    tasks = []
    for path in sorted(root.glob("*_v[0-9]*.py")):
        match = re.fullmatch(r"(.+)_v(\d+)", path.stem)
        assert match is not None
        name, version = match.groups()
        node = ast.parse((root / name / f"{name}.py").read_text())
        cls = next(
            n
            for n in node.body
            if isinstance(n, ast.ClassDef) and n.name == "raw_env"
        )
        init = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        defaults = {
            arg.arg: ast.literal_eval(value)
            for arg, value in zip(
                init.args.args[-len(init.args.defaults) :],
                init.args.defaults,
                strict=True,
            )
        }
        title = "".join(word.title() for word in name.split("_"))
        tasks.append((name, version, title, defaults))
    lines = [
        "// Generated from MPE2 v1.1.1; do not edit.",
        "#ifndef THIRD_PARTY_MPE2_TASKS_H_",
        "#define THIRD_PARTY_MPE2_TASKS_H_",
        "#include <array>",
        "namespace mpe2 {",
        "enum class Task {" + ",".join("k" + t[2] for t in tasks) + "};",
        "struct TaskInfo { Task task; const char* name; const char* id; const char* alias; std::array<int, 9> defaults; };",
        f"inline constexpr std::array<TaskInfo, {len(tasks)}> kTasks = {{{{",
    ]
    for name, version, title, defaults in tasks:
        values = ",".join(str(defaults.get(field, 0)) for field in fields)
        lines.append(
            f'{{Task::k{title}, "{name}", "MPE2/{title}-v{version}", "mpe2/{name}_v{version}", {{{values}}}}},'
        )
    lines.extend(["}};", "}  // namespace mpe2", "#endif"])
    args.output.write_text("\n".join(lines) + "\n")
    font = (root / "_mpe_utils" / "secrcode.ttf").read_bytes()
    args.font_output.write_text(
        "// Secret Code font, copyright Matthew Welch; see FONT_LICENSE.\n"
        "#ifndef THIRD_PARTY_MPE2_FONT_H_\n#define THIRD_PARTY_MPE2_FONT_H_\n"
        "namespace mpe2 {\ninline constexpr unsigned char kFont[] = {\n"
        + ",".join(str(b) for b in font)
        + "\n};\n}\n#endif\n"
    )


if __name__ == "__main__":
    main()
