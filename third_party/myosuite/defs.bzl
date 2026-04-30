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

"""Runtime asset assembler for pinned MyoSuite sources."""

_MYODM_OBJECTS = [
    "airplane",
    "alarmclock",
    "apple",
    "banana",
    "binoculars",
    "bowl",
    "camera",
    "coffeemug",
    "cubelarge",
    "cubemedium",
    "cubesmall",
    "cup",
    "cylinderlarge",
    "cylindermedium",
    "cylindersmall",
    "duck",
    "elephant",
    "eyeglasses",
    "flashlight",
    "flute",
    "gamecontroller",
    "hammer",
    "hand",
    "headphones",
    "knife",
    "lightbulb",
    "mouse",
    "mug",
    "phone",
    "piggybank",
    "pyramidlarge",
    "pyramidmedium",
    "pyramidsmall",
    "scissors",
    "spherelarge",
    "spheremedium",
    "spheresmall",
    "stamp",
    "stanfordbunny",
    "stapler",
    "teapot",
    "toothbrush",
    "toothpaste",
    "toruslarge",
    "torusmedium",
    "torussmall",
    "train",
    "watch",
    "waterbottle",
    "wineglass",
]

def _myosuite_runtime_assets_impl(ctx):
    out = ctx.actions.declare_directory(ctx.attr.out)
    manifest = ctx.actions.declare_file(ctx.label.name + "_srcs.txt")
    srcs = sorted([src.path for src in ctx.files.srcs])
    ctx.actions.write(output = manifest, content = "\n".join(srcs) + "\n")

    object_manifest = ctx.actions.declare_file(ctx.label.name + "_objects.txt")
    ctx.actions.write(output = object_manifest, content = "\n".join(_MYODM_OBJECTS) + "\n")

    args = ctx.actions.args()
    args.add(out.path)
    args.add(manifest.path)
    args.add(object_manifest.path)

    ctx.actions.run_shell(
        inputs = depset(ctx.files.srcs + [manifest, object_manifest]),
        outputs = [out],
        arguments = [args],
        command = r"""
set -eu

out="$1"
manifest="$2"
objects="$3"

mkdir -p "$out/myosuite" "$out/myosuite/simhive"
while IFS= read -r src; do
  [ -n "$src" ] || continue

  case "$src" in
    *myosuite_source*/myosuite/*)
      rel="${src#*myosuite_source*/myosuite/}"
      dst="$out/myosuite/$rel"
      ;;
    *myosuite_mpl_sim*/*)
      rel="${src#*myosuite_mpl_sim*/}"
      dst="$out/myosuite/simhive/MPL_sim/$rel"
      ;;
    *myosuite_ycb_sim*/*)
      rel="${src#*myosuite_ycb_sim*/}"
      dst="$out/myosuite/simhive/YCB_sim/$rel"
      ;;
    *myosuite_furniture_sim*/*)
      rel="${src#*myosuite_furniture_sim*/}"
      dst="$out/myosuite/simhive/furniture_sim/$rel"
      ;;
    *myosuite_myo_sim*/*)
      rel="${src#*myosuite_myo_sim*/}"
      dst="$out/myosuite/simhive/myo_sim/$rel"
      ;;
    *myosuite_object_sim*/*)
      rel="${src#*myosuite_object_sim*/}"
      dst="$out/myosuite/simhive/object_sim/$rel"
      ;;
    *)
      continue
      ;;
  esac

  case "$rel" in
    .gitignore|.idea/*|BUILD.bazel|REPO.bazel|WORKSPACE|README.md|test_sims.py)
      continue
      ;;
    scene/*.mtl|scene/*.obj)
      continue
      ;;
  esac

  mkdir -p "$(dirname "$dst")"
  cp -R "$src" "$dst"
done < "$manifest"

template="$out/myosuite/envs/myo/assets/hand/myohand_object.xml"
while IFS= read -r object_name; do
  [ -n "$object_name" ] || continue
  sed "s/OBJECT_NAME/${object_name}/g" "$template" \
    > "$out/myosuite/envs/myo/assets/hand/myohand_object_${object_name}.xml"
done < "$objects"
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
        "srcs": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
        "out": attr.string(mandatory = True),
    },
)
