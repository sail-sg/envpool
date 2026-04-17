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

"""MyoSuite runtime asset staging rule."""

def _myosuite_runtime_assets_impl(ctx):
    out = ctx.actions.declare_directory(ctx.attr.out)
    manifest = ctx.actions.declare_file(ctx.label.name + "_srcs.txt")
    lines = []

    def add_entries(srcs, strip_prefix, dest_prefix):
        for src in sorted(srcs, key = lambda f: f.path):
            lines.append("\t".join([src.path, strip_prefix, dest_prefix]))

    add_entries(ctx.files.myosuite_env_assets, "myosuite/", "")
    add_entries(ctx.files.myo_sim_assets, "myo_sim_src/", "simhive/myo_sim/")
    add_entries(
        ctx.files.furniture_sim_assets,
        "furniture_sim_src/",
        "simhive/furniture_sim/",
    )
    add_entries(
        ctx.files.object_sim_assets,
        "object_sim_src/",
        "simhive/object_sim/",
    )
    add_entries(ctx.files.mpl_sim_assets, "mpl_sim_src/", "simhive/MPL_sim/")
    add_entries(ctx.files.ycb_sim_assets, "ycb_sim_src/", "simhive/YCB_sim/")
    add_entries(ctx.files.patched_assets, "third_party/myosuite/", "")

    ctx.actions.write(output = manifest, content = "\n".join(lines) + "\n")

    args = ctx.actions.args()
    args.add(out.path)
    args.add(manifest.path)

    ctx.actions.run_shell(
        inputs = depset(
            ctx.files.myosuite_env_assets +
            ctx.files.myo_sim_assets +
            ctx.files.furniture_sim_assets +
            ctx.files.object_sim_assets +
            ctx.files.mpl_sim_assets +
            ctx.files.ycb_sim_assets +
            ctx.files.patched_assets +
            [manifest],
        ),
        outputs = [out],
        arguments = [args],
        command = """
set -eu

out="$1"
manifest="$2"

mkdir -p "$out"
while IFS=$'\\t' read -r src strip_prefix dest_prefix; do
  [ -n "$src" ] || continue
  rel="${src#*${strip_prefix}}"
  if [ "$rel" = "$src" ]; then
    rel="$(basename "$src")"
  fi
  dst="$out/$dest_prefix$rel"
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

myosuite_runtime_assets = rule(
    implementation = _myosuite_runtime_assets_impl,
    attrs = {
        "myosuite_env_assets": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
        "myo_sim_assets": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
        "furniture_sim_assets": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
        "object_sim_assets": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
        "mpl_sim_assets": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
        "ycb_sim_assets": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
        "patched_assets": attr.label_list(
            allow_files = True,
        ),
        "out": attr.string(mandatory = True),
    },
)
