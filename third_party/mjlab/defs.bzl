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

"""Declare a bounded set of native source shards, with generated task coverage."""

def _recipe_args(ctx, args):
    for file in ctx.files.recipe:
        args.add("--recipe")
        args.add(file.short_path)
        args.add(file.path)

def _native_assets_impl(ctx):
    assets = ctx.actions.declare_directory("generated/assets")
    testdata = ctx.actions.declare_directory("generated/testdata")
    header = ctx.actions.declare_file("generated/registry.h")
    registry = ctx.actions.declare_file("generated/registry.json")
    versions = ctx.actions.declare_file("generated/versions.json")
    sources = [ctx.actions.declare_file("generated/kernels_" + str(i) + ".cc") for i in range(16)]
    sources.append(ctx.actions.declare_file("generated/lookup.cc"))
    warp_headers = [ctx.actions.declare_file("warp_headers/" + name) for name in ["exports.h", "version.h"]]
    outputs = [assets, testdata, header, registry, versions] + sources + warp_headers
    if ctx.file.archive:
        args = ctx.actions.args()
        args.add("unpack")
        args.add("--archive", ctx.file.archive)
        args.add("--output", ctx.bin_dir.path + "/" + ctx.label.package)
        _recipe_args(ctx, args)
        ctx.actions.run(
            executable = ctx.executable.codegen,
            inputs = [ctx.file.archive] + ctx.files.recipe,
            outputs = outputs,
            arguments = [args],
            mnemonic = "UnpackMjlabCodegen",
        )
    else:
        raw_sources = ctx.actions.declare_directory("generated/cpp")
        args = ctx.actions.args()
        args.add("--output", header.dirname)
        ctx.actions.run(
            executable = ctx.executable.generator,
            outputs = [assets, testdata, registry, versions, raw_sources],
            arguments = [args],
            env = {"WANDB_MODE": "disabled", "MJLAB_WARP_QUIET": "1"},
            mnemonic = "GenerateMjlabNativeAssets",
            progress_message = "Exporting all pinned MJLab tasks and native CPU kernels",
            use_default_shell_env = True,
        )
        args = ctx.actions.args()
        args.add("--source", raw_sources.path)
        args.add("--registry", registry.path)
        args.add("--output", header.dirname)
        ctx.actions.run(
            executable = ctx.executable.linker,
            inputs = [raw_sources, registry],
            outputs = [header] + sources,
            arguments = [args],
            mnemonic = "LinkMjlabNativeSources",
        )
        ctx.actions.run(
            executable = ctx.executable.header_generator,
            outputs = warp_headers,
            arguments = [file.path for file in warp_headers],
            mnemonic = "GenerateMjlabWarpHeaders",
        )
    return [
        DefaultInfo(files = depset(outputs)),
        OutputGroupInfo(
            assets = depset([assets, registry, versions]),
            testdata = depset([testdata]),
            sources = depset(sources),
            headers = depset([header]),
            registry = depset([registry]),
            warp_headers = depset(warp_headers),
        ),
    ]

native_assets = rule(
    implementation = _native_assets_impl,
    attrs = {
        "archive": attr.label(allow_single_file = [".tar.gz"]),
        "codegen": attr.label(
            default = "//third_party/mjlab:codegen",
            executable = True,
            cfg = "exec",
        ),
        "recipe": attr.label(default = "//third_party/mjlab:codegen_recipe"),
        "generator": attr.label(
            default = "//third_party/mjlab:generate",
            executable = True,
            cfg = "exec",
        ),
        "linker": attr.label(
            default = "//third_party/mjlab:link",
            executable = True,
            cfg = "exec",
        ),
        "header_generator": attr.label(
            default = "//third_party/mjlab:generate_headers",
            executable = True,
            cfg = "exec",
        ),
    },
)

def _codegen_bundle_impl(ctx):
    output = ctx.actions.declare_file("codegen/codegen_input.tar.gz")
    args = ctx.actions.args()
    args.add("pack")
    args.add("--archive", output)
    _recipe_args(ctx, args)
    for file in ctx.files.srcs:
        args.add("--payload")
        args.add(file.short_path[len(ctx.label.package) + 1:])
        args.add(file.path)
    ctx.actions.run(
        executable = ctx.executable.codegen,
        inputs = ctx.files.srcs + ctx.files.recipe,
        outputs = [output],
        arguments = [args],
        mnemonic = "PackMjlabCodegen",
    )
    return [DefaultInfo(files = depset([output]))]

codegen_bundle = rule(
    implementation = _codegen_bundle_impl,
    attrs = {
        "srcs": attr.label_list(mandatory = True),
        "recipe": attr.label(default = "//third_party/mjlab:codegen_recipe"),
        "codegen": attr.label(
            default = "//third_party/mjlab:codegen",
            executable = True,
            cfg = "exec",
        ),
    },
)

# Native runtime libraries come from the pinned oracle wheel, regardless of
# the interpreter ABI used to build the EnvPool extension.
_ORACLE_PYTHON_FLAG = str(Label("@rules_python//python/config_settings:python_version"))

def _oracle_python_impl(_settings, _attr):
    return {_ORACLE_PYTHON_FLAG: "3.12"}

_oracle_python = transition(
    implementation = _oracle_python_impl,
    inputs = [],
    outputs = [_ORACLE_PYTHON_FLAG],
)

def _wheel_files_impl(ctx):
    outputs = ctx.outputs.outs
    args = ctx.actions.args()
    args.add(ctx.file.wheel)
    for member, file in zip(ctx.attr.members, outputs):
        args.add(member)
        args.add(file)
    ctx.actions.run(
        executable = ctx.executable._extractor,
        inputs = [ctx.file.wheel],
        outputs = outputs,
        arguments = [args],
        mnemonic = "ExtractMjlabMathRuntime",
    )
    return [DefaultInfo(files = depset(outputs))]

_wheel_files = rule(
    implementation = _wheel_files_impl,
    attrs = {
        "wheel": attr.label(
            allow_single_file = [".whl"],
            cfg = _oracle_python,
            mandatory = True,
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
        "members": attr.string_list(mandatory = True),
        "outs": attr.output_list(mandatory = True),
        "_extractor": attr.label(
            default = "//third_party/mjlab:extract_wheel",
            executable = True,
            cfg = "exec",
        ),
    },
)

def wheel_files(name, files, **kwargs):
    """Extract named native runtime files without importing the wheel package."""
    _wheel_files(
        name = name,
        members = files.values(),
        outs = files.keys(),
        **kwargs
    )
