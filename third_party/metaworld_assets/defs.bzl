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

"""Runtime asset copier for MetaWorld."""

def _metaworld_runtime_assets_impl(ctx):
    out = ctx.actions.declare_directory(ctx.attr.out)
    manifest = ctx.actions.declare_file(ctx.label.name + "_srcs.txt")
    srcs = sorted([src.path for src in ctx.files.srcs])
    ctx.actions.write(output = manifest, content = "\n".join(srcs) + "\n")

    args = ctx.actions.args()
    args.add(out.path)
    args.add(ctx.attr.strip_prefix)
    args.add(manifest.path)

    ctx.actions.run_shell(
        inputs = depset(ctx.files.srcs + [manifest]),
        outputs = [out],
        arguments = [args],
        command = r"""
set -eu

out="$1"
strip_prefix="$2"
manifest="$3"

mkdir -p "$out"
while IFS= read -r src; do
  [ -n "$src" ] || continue
  rel="${src#*${strip_prefix}}"
  if [ "$rel" = "$src" ]; then
    rel="$(basename "$src")"
  fi

  dst="$out/$rel"
  mkdir -p "$(dirname "$dst")"
  cp -R "$src" "$dst"
done < "$manifest"
""",
    )

    return [
        DefaultInfo(
            files = depset([out]),
            runfiles = ctx.runfiles(files = [out]),
        ),
    ]

metaworld_runtime_assets = rule(
    implementation = _metaworld_runtime_assets_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
        "out": attr.string(mandatory = True),
        "strip_prefix": attr.string(mandatory = True),
    },
)
