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

load("@mjl//:requirements.bzl", "all_requirements", "requirement")
load("@python_versions//3.12:defs.bzl", py_binary_312 = "py_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load(":defs.bzl", "codegen_bundle", "native_assets")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "macos_arm64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
)

exports_files([
    "mjlab.BUILD",
    "mkl.BUILD.tpl",
    "sleef.BUILD",
    "oracle_requirements.txt",
    "warp.BUILD",
    "warp_native.patch",
])

py_library(
    name = "oracle",
    srcs = ["oracle_util.py"],
    imports = ["../.."],
    deps = all_requirements + ["@mjlab_source//:python"],
)

py_binary_312(
    name = "generate",
    srcs = [
        "capture.py",
        "generate.py",
        "motion_fixture.py",
        "pack.py",
    ],
    imports = ["."],
    main = "generate.py",
    deps = [":oracle"],
)

py_binary(
    name = "link",
    srcs = [
        "aot.py",
        "link.py",
    ],
    imports = ["."],
    main = "link.py",
)

py_binary_312(
    name = "codegen",
    srcs = ["codegen.py"],
)

filegroup(
    name = "codegen_recipe",
    srcs = ["BUILD"] + glob([
        "*.BUILD",
        "*.bzl",
        "*.patch",
        "*.py",
        "*.txt",
    ]),
)

# This optional input is supplied by the release codegen job.
# buildifier: disable=constant-glob
_CODEGEN_INPUT = glob(["codegen_input.tar.gz"])

native_assets(
    name = "generate_native_assets",
    archive = _CODEGEN_INPUT[0] if _CODEGEN_INPUT else None,
)

codegen_bundle(
    name = "codegen_bundle",
    srcs = [":generate_native_assets"],
    tags = ["manual"],
)

filegroup(
    name = "assets",
    srcs = [":generate_native_assets"],
    output_group = "assets",
)

filegroup(
    name = "testdata",
    srcs = [":generate_native_assets"],
    output_group = "testdata",
)

filegroup(
    name = "sources",
    srcs = [":generate_native_assets"],
    output_group = "sources",
)

filegroup(
    name = "headers",
    srcs = [":generate_native_assets"],
    output_group = "headers",
)

cc_library(
    name = "metadata",
    hdrs = [":headers"],
)

cc_library(
    name = "kernels",
    srcs = [":sources"],
    # Warp's spatial.h exposes overlapping vec3/vec6 views. Type-based alias
    # analysis can discard gyro values with GCC 14 on ARM. Preserve those
    # intended aliases in generated CPU code, without changing its arithmetic.
    copts = select({
        "//:windows": [
            "/fp:strict",
            "/bigobj",
        ],
        ":macos_arm64": [
            "-fno-strict-aliasing",
            "-ffp-contract=off",
            "-mcpu=generic",
        ],
        "//conditions:default": [
            "-fno-strict-aliasing",
            "-ffp-contract=off",
            # Keep the upstream CPU JIT's separate calls. GCC combines them
            # into sincosf, which rounds differently on glibc 2.28 x64.
            "-fno-builtin-sinf",
            "-fno-builtin-cosf",
        ],
    }),
    deps = ["@mjlab_warp//:native"],
)

py_binary_312(
    name = "generate_headers",
    srcs = ["generate_headers.py"],
    deps = [requirement("warp-lang")],
)

filegroup(
    name = "warp_headers",
    srcs = [":generate_native_assets"],
    output_group = "warp_headers",
)

cc_library(
    name = "warp_generated_headers",
    hdrs = [":warp_headers"],
    includes = ["warp_headers"],
)

filegroup(
    name = "licenses",
    srcs = [
        "AUTHORS.mujoco-warp",
        "LICENSE.isaaclab",
        "LICENSE.mujoco-warp",
        "LICENSE.pytorch",
        "NOTICE",
        "@mjlab_mkl//:licenses",
        "@mjlab_source//:licenses",
        "@mjlab_warp//:licenses",
    ] + select({
        "@platforms//cpu:arm64": ["@mjlab_sleef//:LICENSE.txt"],
        "//conditions:default": [],
    }),
)
