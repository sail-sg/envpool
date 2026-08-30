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
"""Give pinned Warp CPU modules distinct native linkage, without a JIT."""

import re

_PREAMBLE = "#define builtin_block_dim() wp::block_dim()\n"
_ENTRY = re.compile(
    r"WP_API void (\w+_cpu_forward)\(\s*"
    r"(wp::launch_bounds_t<\d+>) \*dim,\s*"
    r"(wp_args_\w+) \*_wp_args\)"
)


def native_module(source: str, module_hash: str) -> tuple[str, dict[str, str]]:
    """Wrap one generated translation unit and expose its typed CPU entries.

    JIT modules occupy separate symbol tables. Native modules share one binary,
    so identical helper names and progressively specialized generic modules
    must remain separate here too. Only linkage changes; kernel bodies do not.
    Refuse an unexpected upstream source format instead of losing an entry.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", module_hash):
        raise ValueError("expected a full Warp module hash")
    if source.count(_PREAMBLE) != 1:
        raise ValueError("unexpected Warp CPU module preamble")
    entries = _ENTRY.findall(source)
    exported = re.findall(r"WP_API void (\w+_cpu_forward)\(", source)
    if not entries or len(entries) != len(exported):
        raise ValueError("unexpected Warp CPU entry signature")
    namespace = "mjlab_" + module_hash
    preamble, body = source.split(_PREAMBLE)
    body = body.replace('extern "C" {', "").replace("} // extern C", "")
    wrappers = []
    symbols = {}
    for name, bounds, args in entries:
        exported_name = namespace + "_" + name
        symbols[name] = exported_name
        wrappers.append(
            f'extern "C" void {exported_name}(void* dim, void* args) {{\n'
            f"  {namespace}::{name}(static_cast<{bounds}*>(dim),\n"
            f"      static_cast<{namespace}::{args}*>(args));\n"
            "}\n"
        )
    macros = sorted(
        set(re.findall(r"^#define\s+(\w+)\(", source, re.MULTILINE))
    )
    return (
        preamble
        + _PREAMBLE
        + f"\nnamespace {namespace} {{\n"
        + body
        + f"\n}}  // namespace {namespace}\n"
        + "\n".join(wrappers)
        # Each JIT module originally has its own preprocessor state. The
        # function-like int/float macros must not leak into the next shard's
        # standard-library includes when several modules share a native TU.
        + "\n".join(f"#undef {name}" for name in macros)
        + "\n",
        symbols,
    )
