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

    ctx.actions.run(
        executable = ctx.executable.generator,
        inputs = depset(ctx.files.srcs + [manifest, object_manifest]),
        outputs = [out],
        arguments = [args],
        env = {
            "PATH": "/usr/sbin:/usr/bin:/bin:/opt/homebrew/bin:/usr/local/bin",
        },
        mnemonic = "GenerateMyoSuiteRuntimeAssets",
        progress_message = "Generating native MyoSuite runtime assets",
        use_default_shell_env = True,
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
        "generator": attr.label(
            default = "//third_party/myosuite:generate_runtime_assets",
            executable = True,
            cfg = "exec",
        ),
    },
)

def _write_manifest(ctx, name, files):
    manifest = ctx.actions.declare_file(ctx.label.name + "_" + name + ".txt")
    ctx.actions.write(
        output = manifest,
        content = "\n".join(sorted([src.path for src in files])) + "\n",
    )
    return manifest

def _myosuite_native_assets_impl(ctx):
    tasks_json = ctx.actions.declare_file("myosuite_tasks.json")
    tasks_header = ctx.actions.declare_file("myosuite_tasks.h")
    tasks_python = ctx.actions.declare_file("myosuite_generated_tasks.py")
    metadata_json = ctx.actions.declare_file("myosuite_task_metadata.json")
    metadata_header = ctx.actions.declare_file("myosuite_task_metadata.h")
    reference_header = ctx.actions.declare_file("myosuite_reference_data.h")

    myosuite_manifest = _write_manifest(ctx, "myosuite", ctx.files.myosuite_srcs)
    mpl_manifest = _write_manifest(ctx, "mpl", ctx.files.mpl_srcs)
    ycb_manifest = _write_manifest(ctx, "ycb", ctx.files.ycb_srcs)
    furniture_manifest = _write_manifest(ctx, "furniture", ctx.files.furniture_srcs)
    myo_manifest = _write_manifest(ctx, "myo", ctx.files.myo_srcs)
    object_manifest = _write_manifest(ctx, "object", ctx.files.object_srcs)

    args = ctx.actions.args()
    args.add("--myosuite-manifest", myosuite_manifest)
    args.add("--mpl-manifest", mpl_manifest)
    args.add("--ycb-manifest", ycb_manifest)
    args.add("--furniture-manifest", furniture_manifest)
    args.add("--myo-manifest", myo_manifest)
    args.add("--object-manifest", object_manifest)
    args.add("--out-tasks-json", tasks_json)
    args.add("--out-tasks-header", tasks_header)
    args.add("--out-tasks-python", tasks_python)
    args.add("--out-metadata-json", metadata_json)
    args.add("--out-metadata-header", metadata_header)
    args.add("--out-reference-header", reference_header)

    manifests = [
        myosuite_manifest,
        mpl_manifest,
        ycb_manifest,
        furniture_manifest,
        myo_manifest,
        object_manifest,
    ]
    ctx.actions.run(
        executable = ctx.executable.generator,
        inputs = depset(
            ctx.files.myosuite_srcs +
            ctx.files.mpl_srcs +
            ctx.files.ycb_srcs +
            ctx.files.furniture_srcs +
            ctx.files.myo_srcs +
            ctx.files.object_srcs +
            manifests,
        ),
        outputs = [
            tasks_json,
            tasks_header,
            tasks_python,
            metadata_json,
            metadata_header,
            reference_header,
        ],
        arguments = [args],
        env = {
            "PATH": "/usr/sbin:/usr/bin:/bin:/opt/homebrew/bin:/usr/local/bin",
        },
        mnemonic = "GenerateMyoSuiteNativeAssets",
        progress_message = "Generating native MyoSuite metadata from pinned upstream source",
        use_default_shell_env = True,
    )

    headers = depset([tasks_header, metadata_header, reference_header])
    json_files = depset([tasks_json, metadata_json])
    python_files = depset([tasks_python])
    all_files = depset([
        tasks_json,
        tasks_header,
        tasks_python,
        metadata_json,
        metadata_header,
        reference_header,
    ])
    return [
        DefaultInfo(files = all_files),
        OutputGroupInfo(
            headers = headers,
            json = json_files,
            python = python_files,
        ),
    ]

myosuite_native_assets = rule(
    implementation = _myosuite_native_assets_impl,
    attrs = {
        "myosuite_srcs": attr.label_list(allow_files = True, mandatory = True),
        "mpl_srcs": attr.label_list(allow_files = True, mandatory = True),
        "ycb_srcs": attr.label_list(allow_files = True, mandatory = True),
        "furniture_srcs": attr.label_list(allow_files = True, mandatory = True),
        "myo_srcs": attr.label_list(allow_files = True, mandatory = True),
        "object_srcs": attr.label_list(allow_files = True, mandatory = True),
        "generator": attr.label(
            executable = True,
            cfg = "exec",
            mandatory = True,
        ),
    },
)
