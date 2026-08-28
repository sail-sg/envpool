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

"""Extract the maze data from the pinned Jumanji source without importing it."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


def main() -> None:
    """Emit a C++ header containing only the official maze data."""
    source, output = map(Path, sys.argv[1:])
    tree = ast.parse(source.read_text())
    maze = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_MAZE"
            for target in node.targets
        )
    )
    rows = ",\n".join("    " + json.dumps(row) for row in maze)
    output.write_text(
        "// Generated from Jumanji 1.1.2 (Apache-2.0). Do not edit.\n"
        "#ifndef THIRD_PARTY_JUMANJI_PACMAN_MAZE_H_\n"
        "#define THIRD_PARTY_JUMANJI_PACMAN_MAZE_H_\n"
        "namespace jumanji { namespace pacman {\n"
        "inline constexpr const char* kMaze[] = {\n" + rows + "};\n"
        "}}  // namespace jumanji::pacman\n"
        "#endif  // THIRD_PARTY_JUMANJI_PACMAN_MAZE_H_\n"
    )


if __name__ == "__main__":
    main()
