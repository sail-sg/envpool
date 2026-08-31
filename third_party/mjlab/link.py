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
"""Give exported CPU kernels native linkage and emit stable build outputs."""

import argparse
import json
from pathlib import Path

from aot import native_module


def main() -> None:
    """Write native source shards and registry metadata for Bazel."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    registry = json.dumps(
        json.loads(args.registry.read_text()), separators=(",", ":")
    )
    (args.output / "registry.h").write_text(
        "#pragma once\nnamespace mjlab {\n"
        f'inline constexpr char kRegistry[] = R"mjlab({registry})mjlab";\n'
        "}  // namespace mjlab\n"
    )
    modules, symbols = [], {}
    for path in sorted(args.source.glob("*.cc")):
        module, entries = native_module(path.read_text(), path.stem)
        modules.append(module)
        symbols.update({
            path.stem + ":" + name.removesuffix("_cpu_forward"): value
            for name, value in entries.items()
        })
    declarations = "\n".join(
        f'extern "C" void {name}(void*, void*);'
        for name in sorted(set(symbols.values()))
    )
    entries = "\n".join(
        f"    {{{json.dumps(key)}, &{name}}},"
        for key, name in sorted(symbols.items())
    )
    lookup = (
        "#include <string>\n#include <unordered_map>\n"
        + declarations
        + "\nnamespace mjlab {\nusing Kernel = void (*)(void*, void*);\n"
        + "Kernel LookupKernel(const std::string& key) {\n"
        + "  static const std::unordered_map<std::string, Kernel> kernels = {\n"
        + entries
        + "\n  };\n  const auto it = kernels.find(key);\n"
        + "  return it == kernels.end() ? nullptr : it->second;\n}\n}\n"
    )
    # Keep the ordinary C++ lookup table separate from generated kernel shards.
    (args.output / "lookup.cc").write_text(lookup)
    for shard in range(16):
        (args.output / f"kernels_{shard}.cc").write_text(
            "\n".join(modules[shard::16])
        )


if __name__ == "__main__":
    main()
