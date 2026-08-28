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
"""Embed only PNG files referenced by the pinned upstream texture loaders."""

import argparse
import ast
from pathlib import Path


def main() -> None:
    """Generate the requested build artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classic", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lines = [
        "// Generated from Craftax v1.6.1 assets; MIT license.",
        '#include "envpool/craftax/renderer.h"',
        "namespace craftax {",
    ]
    tables = []
    blobs: dict[bytes, str] = {}
    for family, source in (("classic", args.classic), ("full", args.full)):
        tree = ast.parse(source.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "load_all_textures_given_size"
        )
        filenames = sorted({
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.endswith(".png")
        })
        for filename in filenames:
            data = (source.parent / "assets" / filename).read_bytes()
            if data not in blobs:
                name = f"png_{len(blobs)}"
                blobs[data] = name
                lines.append(
                    f"static const unsigned char {name}[] = {{"
                    + ",".join(map(str, data))
                    + "};"
                )
            tables.append(
                f'  {{"{family}/{filename}", {blobs[data]}, {len(data)}}},'
            )
    lines.extend([
        "const EncodedTexture kEncodedTextures[] = {",
        *tables,
        "};",
        "const EncodedTexture* const kTextures = kEncodedTextures;",
        "const std::size_t kTextureCount = sizeof(kEncodedTextures) / sizeof(kEncodedTextures[0]);",
        "}  // namespace craftax",
        "",
    ])
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
