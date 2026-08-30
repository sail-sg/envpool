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

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@python_versions//3.12:defs.bzl", py_binary_312 = "py_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//envpool:requirements.bzl", "requirement")
load("//third_party:common.bzl", "copy_to_directory")

package(default_visibility = ["//visibility:public"])

copy_to_directory(
    name = "assets",
    srcs = ["//third_party/mjlab:assets"],
    out = "assets",
    strip_prefix = "third_party/mjlab/generated/assets",
)

copy_to_directory(
    name = "licenses",
    srcs = ["//third_party/mjlab:licenses"],
    out = "licenses",
    # Preserve vendor directories: multiple dependencies use LICENSE.txt.
    strip_prefix = "external/",
)

cc_library(
    name = "simulation",
    srcs = [
        "assets.cc",
        "events.cc",
        "manipulation.cc",
        "math.cc",
        "motion.cc",
        "physics.cc",
        "sensors.cc",
        "simulation.cc",
        "terrain.cc",
        "tracking.cc",
        "velocity.cc",
    ],
    hdrs = [
        "assets.h",
        "math.h",
        "motion.h",
        "physics.h",
        "simulation.h",
        "terrain.h",
    ],
    copts = select({
        "//:windows": ["/fp:strict"],
        "//conditions:default": ["-ffp-contract=off"],
    }),
    data = [":assets"],
    deps = [
        "//third_party/mjlab:kernels",
        "@boost//:json",
        "@mjlab_mkl//:math",
        "@mjlab_warp//:native",
        "@mujoco//:mujoco_lib",
        "@zlib",
        "@zstd",
    ] + select({
        "@platforms//cpu:arm64": ["@mjlab_sleef//:sleef"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "mjlab_env",
    hdrs = ["mjlab_env.h"],
    deps = [
        ":simulation",
        "//envpool/core:async_envpool",
        "//envpool/mujoco:mujoco_render",
        "//third_party/mjlab:metadata",
    ],
)

pybind_extension(
    name = "mjlab_envpool",
    srcs = ["mjlab_envpool.cc"],
    deps = [
        ":mjlab_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "mjlab",
    srcs = ["__init__.py"],
    data = [
        ":assets",
        ":licenses",
        ":mjlab_envpool",
    ],
    imports = ["../../.."],
    deps = [
        "//envpool/python:api",
        "//envpool/python:glfw_context",
        requirement("dm-env"),
        requirement("gymnasium"),
        requirement("numpy"),
    ],
)

py_library(
    name = "registration",
    srcs = ["registration.py"],
    imports = ["../../.."],
    deps = [
        ":mjlab",
        "//envpool:registration",
    ],
)

py_library(
    name = "test_support",
    testonly = True,
    srcs = ["test_support.py"],
    data = ["//third_party/mjlab:testdata"],
    imports = ["../../.."],
    deps = [requirement("numpy")],
)

py_test(
    name = "mjlab_test",
    size = "enormous",
    srcs = ["mjlab_test.py"],
    imports = [
        ".",
        "../../..",
    ],
    shard_count = 4,
    deps = [
        ":registration",
        ":test_support",
        "//envpool:registration",
        "//envpool/mujoco:render_test_utils",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_binary_312(
    name = "oracle_probe",
    testonly = True,
    srcs = ["oracle_probe.py"],
    data = ["@mujoco//:mujoco_shared_lib"],
    imports = ["../../.."],
    deps = [
        "//envpool/mujoco:mujoco_oracle",
        "//envpool/mujoco:mujoco_render_oracle",
        "//envpool/python:glfw_context",
        "//third_party/mjlab:oracle",
    ],
)

_ALIGN_DEPS = [
    ":registration",
    ":test_support",
    "//envpool:registration",
    "//envpool/mujoco:render_test_utils",
    requirement("absl-py"),
    requirement("numpy"),
]

py_binary_312(
    name = "render_doc",
    testonly = True,
    srcs = [
        "mjlab_align_test.py",
        "render_doc.py",
    ],
    data = [":oracle_probe"],
    imports = ["../../.."],
    main = "render_doc.py",
    deps = _ALIGN_DEPS + [requirement("matplotlib")],
)

py_test(
    name = "mjlab_align_test",
    size = "enormous",
    srcs = ["mjlab_align_test.py"],
    data = [":oracle_probe"],
    imports = ["../../.."],
    shard_count = 4,
    deps = _ALIGN_DEPS,
)
