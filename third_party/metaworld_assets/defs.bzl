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
    args.add("1" if ctx.attr.compact else "0")

    ctx.actions.run_shell(
        inputs = depset(ctx.files.srcs + [manifest]),
        outputs = [out],
        arguments = [args],
        command = r"""
set -eu

out="$1"
strip_prefix="$2"
manifest="$3"
compact="$4"

mkdir -p "$out"
while IFS= read -r src; do
  [ -n "$src" ] || continue
  rel="${src#*${strip_prefix}}"
  if [ "$rel" = "$src" ]; then
    rel="$(basename "$src")"
  fi

  if [ "$compact" = "1" ]; then
    case "$rel" in
      textures/wood2.png)
        ;;
      textures/*)
        continue
        ;;
    esac
  fi

  dst="$out/$rel"
  mkdir -p "$(dirname "$dst")"
  if [ "$compact" = "1" ]; then
    case "$rel" in
      scene/*.xml)
        sed -E \
          -e "/<texture[^>]*file=['\"][^'\"]*\\/(floor2|metal)\\.png['\"][^>]*>/d" \
          -e "/<material name=['\"]basic_floor['\"]/s/[[:space:]]texture=\"[^\"]*\"//g" \
          -e "/<material name=['\"]basic_floor['\"]/s/[[:space:]]texture='[^']*'//g" \
          -e "/<material name=['\"]basic_floor['\"]/s/[[:space:]]texrepeat=\"[^\"]*\"//g" \
          -e "/<material name=['\"]basic_floor['\"]/s/[[:space:]]texrepeat='[^']*'//g" \
          -e "/<material name=['\"]wall_metal['\"]/s/[[:space:]]texture=\"[^\"]*\"//g" \
          -e "/<material name=['\"]wall_metal['\"]/s/[[:space:]]texture='[^']*'//g" \
          "$src" > "$dst"
        ;;
      *.xml)
        sed -E \
          -e "/<texture[^>]*file=['\"][^'\"]*\\.(png|jpg|jpeg)['\"][^>]*>/d" \
          -e "s/[[:space:]]texture=\"[^\"]*\"//g" \
          -e "s/[[:space:]]texture='[^']*'//g" \
          -e "s/[[:space:]]texrepeat=\"[^\"]*\"//g" \
          -e "s/[[:space:]]texrepeat='[^']*'//g" \
          -e "s/[[:space:]]texuniform=\"[^\"]*\"//g" \
          -e "s/[[:space:]]texuniform='[^']*'//g" \
          "$src" > "$dst"
        ;;
      *)
        cp -R "$src" "$dst"
        ;;
    esac
  else
    cp -R "$src" "$dst"
  fi
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
        "compact": attr.bool(default = False),
    },
)
