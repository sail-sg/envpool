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
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("//envpool:requirements.bzl", "requirement")
load("//third_party:common.bzl", "copy_to_directory")

package(default_visibility = ["//visibility:public"])

copy_to_directory(
    name = "assets",
    srcs = [
        "//third_party/dmc_locomotion:mocap_runtime",
        "//third_party/dmc_locomotion:tasks.json",
        "@mujoco_dmc_xml//:locomotion_assets",
    ],
    out = "assets_dm_control",
    strip_prefix = "mujoco_dmc_xml/",
)

cc_library(
    name = "simulation",
    srcs = [
        "cmu.cc",
        "maze.cc",
        "mocap.cc",
        "observations.cc",
        "scene.cc",
        "simulation.cc",
        "soccer.cc",
        "soccer_scene.cc",
        "tasks.cc",
        "tracking.cc",
    ],
    hdrs = [
        "mocap.h",
        "scene.h",
        "simulation.h",
    ],
    # NumPy evaluates these expressions as separate ufunc operations. Fusing
    # multiply/add changes terrain geometry and long rollout observations.
    copts = select({
        "//:windows": ["/fp:strict"],
        "//conditions:default": ["-ffp-contract=off"],
    }),
    data = [
        ":assets",
        ":labmaze_assets",
    ],
    deps = [
        ":random",
        "//envpool/mujoco:mujoco_render",
        "//third_party/dmc_locomotion:task_metadata",
        "@labmaze_source//:maze",
        "@lodepng",
        "@mujoco//:mujoco_lib",
        "@pugixml",
    ],
)

cc_library(
    name = "random",
    srcs = ["random.cc"],
    hdrs = ["random.h"],
    # NumPy's legacy distributions use ordinary compiled C expressions, unlike
    # the separate ufunc operations in scene/observation code. Keep the normal
    # platform contraction behavior here. The Linux arm64 NumPy wheel disables
    # contraction, unlike the macOS arm64 wheel; nonzero Uniform bounds expose
    # the difference in gap geometry and Soccer respawns.
    copts = select({
        "//:linux_arm64": ["-ffp-contract=off"],
        "//conditions:default": [],
    }),
)

copy_to_directory(
    name = "labmaze_assets",
    srcs = ["@labmaze_source//:textures"],
    out = "assets_labmaze",
    strip_prefix = "labmaze/assets/",
)

cc_library(
    name = "locomotion_env",
    hdrs = ["locomotion_env.h"],
    deps = [
        ":simulation",
        "//envpool/core:async_envpool",
    ],
)

pybind_extension(
    name = "locomotion_envpool",
    srcs = ["locomotion_envpool.cc"],
    deps = [
        ":locomotion_env",
        "//envpool/core:py_envpool",
        "//third_party/dmc_locomotion:task_metadata",
    ],
)

py_library(
    name = "locomotion",
    srcs = ["__init__.py"],
    data = [
        ":assets",
        ":labmaze_assets",
        ":locomotion_envpool",
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
        ":locomotion",
        "//envpool:registration",
    ],
)

py_test(
    name = "locomotion_test",
    size = "enormous",
    srcs = ["locomotion_test.py"],
    imports = ["../../.."],
    deps = [
        ":registration",
        "//envpool:registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_library(
    name = "test_support",
    testonly = True,
    srcs = ["locomotion_test.py"],
    imports = ["../../.."],
    deps = [
        ":registration",
        "//envpool:registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_library(
    name = "oracle",
    testonly = True,
    srcs = ["oracle.py"],
    data = [
        "//third_party/dmc_locomotion:mocap_oracle",
        "@mujoco//:mujoco_shared_lib",
    ] + select({
        "//:windows": ["@labmaze_source//labmaze/cc/python:_random_maze"],
        "//conditions:default": [],
    }),
    imports = ["../../.."],
    deps = [
        "//envpool/mujoco:mujoco_oracle",
        requirement("dm-control"),
        requirement("h5py"),
        requirement("mujoco"),
        requirement("numpy"),
    ],
)

py_binary(
    name = "render_doc",
    testonly = True,
    srcs = ["render_doc.py"],
    imports = ["../../.."],
    deps = [
        ":oracle",
        ":registration",
        "//envpool:registration",
        "//envpool/mujoco:dmc_render_oracle",
        "//envpool/mujoco:mujoco_oracle",
        "//envpool/python:glfw_context",
        requirement("numpy"),
        requirement("pillow"),
    ],
)

py_test(
    name = "locomotion_align_test",
    size = "enormous",
    timeout = "long",
    srcs = ["locomotion_align_test.py"],
    imports = ["../../.."],
    # Full episodes and all 36 tracking clips include native camera readback.
    # Keep the same checks practical on the software-rendered ARM64 runner.
    shard_count = 4,
    deps = [
        ":oracle",
        ":registration",
        ":test_support",
        "//envpool:registration",
        "//envpool/mujoco:dmc_render_oracle",
        "//envpool/mujoco:mujoco_oracle",
        "//envpool/python:glfw_context",
        requirement("absl-py"),
        requirement("dm-control"),
        requirement("h5py"),
        requirement("mujoco"),
        requirement("numpy"),
        requirement("pillow"),
    ],
)
