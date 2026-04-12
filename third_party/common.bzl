# Copyright 2021 Garena Online Private Limited
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

"""Rule for simple expansion of template files.

This performs a simple search over the template file for the keys in
substitutions, and replaces them with the corresponding values.

Typical usage:
::

    load("/tools/build_rules/template_rule", "expand_header_template")
    template_rule(
        name = "ExpandMyTemplate",
        src = "my.template",
        out = "my.txt",
        substitutions = {
          "$VAR1": "foo",
          "$VAR2": "bar",
        }
    )

Args:
    name: The name of the rule.
    template: The template file to expand.
    out: The destination of the expanded file.
    substitutions: A dictionary mapping strings to their substitutions.
"""

def template_rule_impl(ctx):
    """Helper function for template_rule."""
    ctx.actions.expand_template(
        template = ctx.file.src,
        output = ctx.outputs.out,
        substitutions = ctx.attr.substitutions,
    )

template_rule = rule(
    attrs = {
        "src": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "substitutions": attr.string_dict(mandatory = True),
        "out": attr.output(mandatory = True),
    },
    # output_to_genfiles is required for header files.
    output_to_genfiles = True,
    implementation = template_rule_impl,
)

def _copy_to_directory_impl(ctx):
    out = ctx.actions.declare_directory(ctx.attr.out)
    manifest = ctx.actions.declare_file(ctx.label.name + "_srcs.txt")
    strip_prefix = ctx.attr.strip_prefix
    flatten = "1" if ctx.attr.flatten else "0"
    srcs = sorted([src.path for src in ctx.files.srcs])

    ctx.actions.write(output = manifest, content = "\n".join(srcs) + "\n")

    args = ctx.actions.args()
    args.add(out.path)
    args.add(strip_prefix)
    args.add(flatten)
    args.add(manifest.path)

    ctx.actions.run_shell(
        inputs = depset(ctx.files.srcs + [manifest]),
        outputs = [out],
        arguments = [args],
        command = """
set -eu

out="$1"
strip_prefix="$2"
flatten="$3"
manifest="$4"

mkdir -p "$out"
while IFS= read -r src; do
  [ -n "$src" ] || continue
  if [ "$flatten" = "1" ]; then
    rel="$(basename "$src")"
  else
    rel="${src#*${strip_prefix}}"
    if [ "$rel" = "$src" ]; then
      rel="$(basename "$src")"
    fi
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

copy_to_directory = rule(
    implementation = _copy_to_directory_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
        "out": attr.string(mandatory = True),
        "strip_prefix": attr.string(mandatory = True),
        "flatten": attr.bool(default = False),
    },
)

def _copy_files_to_directory_impl(ctx):
    manifest = ctx.actions.declare_file(ctx.label.name + "_srcs.txt")
    strip_prefix = ctx.attr.strip_prefix
    outputs = []
    manifest_lines = []

    for src in sorted(ctx.files.srcs, key = lambda f: f.path):
        if src.is_directory:
            fail("copy_files_to_directory only supports file inputs")
        if ctx.attr.flatten:
            rel = src.path.split("/")[-1]
        else:
            idx = src.path.find(strip_prefix)
            if idx == -1:
                rel = src.path.split("/")[-1]
            else:
                rel = src.path[idx + len(strip_prefix):]
        out = ctx.actions.declare_file(ctx.attr.out + "/" + rel)
        outputs.append(out)
        manifest_lines.append(src.path + "\t" + out.path)

    ctx.actions.write(output = manifest, content = "\n".join(manifest_lines) + "\n")

    args = ctx.actions.args()
    args.add(manifest.path)

    ctx.actions.run_shell(
        inputs = depset(ctx.files.srcs + [manifest]),
        outputs = outputs,
        arguments = [args],
        command = """
set -eu

manifest="$1"

while IFS=$'\\t' read -r src dst; do
  [ -n "$src" ] || continue
  mkdir -p "$(dirname "$dst")"
  cp -R "$src" "$dst"
done < "$manifest"
""",
    )

    return [
        DefaultInfo(
            files = depset(outputs),
            runfiles = ctx.runfiles(files = outputs),
        ),
    ]

copy_files_to_directory = rule(
    implementation = _copy_files_to_directory_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
        "out": attr.string(mandatory = True),
        "strip_prefix": attr.string(mandatory = True),
        "flatten": attr.bool(default = False),
    },
)
