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

def _native_assets_impl(ctx):
    assets = ctx.actions.declare_directory("generated/assets")
    testdata = ctx.actions.declare_directory("generated/testdata")
    header = ctx.actions.declare_file("generated/registry.h")
    registry = ctx.actions.declare_file("generated/registry.json")
    versions = ctx.actions.declare_file("generated/versions.json")
    raw_sources = ctx.actions.declare_directory("generated/cpp")
    sources = [ctx.actions.declare_file("generated/kernels_" + str(i) + ".cc") for i in range(16)]
    sources.append(ctx.actions.declare_file("generated/lookup.cc"))
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
    return [
        DefaultInfo(files = depset([assets, testdata, header, registry, versions] + sources)),
        OutputGroupInfo(
            assets = depset([assets, registry, versions]),
            testdata = depset([testdata]),
            sources = depset(sources),
            headers = depset([header]),
            registry = depset([registry]),
        ),
    ]

native_assets = rule(
    implementation = _native_assets_impl,
    attrs = {
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
    },
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
        "wheel": attr.label(allow_single_file = [".whl"], mandatory = True),
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
