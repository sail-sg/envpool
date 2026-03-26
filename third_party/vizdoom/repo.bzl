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

"""Custom repository rule for ViZDoom patching."""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "patch", "workspace_and_buildfile")

_VIZDOOM_PATCH_TARGETS = [
    "src/c_console.cpp",
    "src/fragglescript/t_func.cpp",
    "src/g_level.cpp",
    "src/s_sound.h",
    "src/viz_message_queue.cpp",
    "src/viz_shared_memory.cpp",
    "src/win32/i_system.cpp",
    "src/win32/i_system.h",
]

def _normalize_line_endings(ctx):
    python = ctx.which("python3") or ctx.which("python")
    if not python:
        fail("python is required to normalize ViZDoom line endings before patching")

    script = """
from pathlib import Path

for rel in {paths}:
    path = Path(rel)
    path.write_bytes(path.read_bytes().replace(b"\\r\\n", b"\\n"))
""".format(paths = repr(_VIZDOOM_PATCH_TARGETS))
    result = ctx.execute([str(python), "-c", script])
    if result.return_code:
        fail("Error normalizing ViZDoom line endings:\\n{}{}".format(result.stderr, result.stdout))

def _vizdoom_archive_impl(ctx):
    ctx.download_and_extract(
        ctx.attr.urls,
        "",
        ctx.attr.sha256,
        ctx.attr.type,
        ctx.attr.strip_prefix,
    )
    workspace_and_buildfile(ctx)
    _normalize_line_endings(ctx)
    patch(ctx)

vizdoom_archive = repository_rule(
    implementation = _vizdoom_archive_impl,
    attrs = {
        "build_file": attr.label(allow_single_file = True),
        "build_file_content": attr.string(),
        "patch_args": attr.string_list(default = []),
        "patch_cmds": attr.string_list(default = []),
        "patch_cmds_win": attr.string_list(default = []),
        "patch_strip": attr.int(default = 0),
        "patch_tool": attr.string(default = ""),
        "patches": attr.label_list(default = []),
        "sha256": attr.string(mandatory = True),
        "strip_prefix": attr.string(default = ""),
        "type": attr.string(default = ""),
        "urls": attr.string_list(mandatory = True),
    },
)
