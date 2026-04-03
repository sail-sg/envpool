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

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//envpool:requirements.bzl", "requirement")
load("//third_party:common.bzl", "copy_to_directory")

package(default_visibility = ["//visibility:public"])

copy_to_directory(
    name = "gen_gymnasium_robotics_assets",
    srcs = ["@gymnasium_robotics_assets"],
    out = "assets",
    strip_prefix = "gymnasium_robotics_assets/assets/",
)

cc_library(
    name = "gymnasium_robotics_env",
    srcs = ["utils.cc"],
    hdrs = [
        "adroit.h",
        "fetch.h",
        "hand.h",
        "kitchen.h",
        "mujoco_env.h",
        "point_maze.h",
        "utils.h",
    ],
    data = [
        ":gen_gymnasium_robotics_assets",
    ],
    deps = [
        "//envpool/core:async_envpool",
        "//envpool/mujoco:mujoco_render",
        "@mujoco//:mujoco_lib",
    ],
)

pybind_extension(
    name = "gymnasium_robotics_envpool",
    srcs = ["gymnasium_robotics_envpool.cc"],
    deps = [
        ":gymnasium_robotics_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "gymnasium_robotics",
    srcs = ["__init__.py"],
    data = [
        ":gen_gymnasium_robotics_assets",
        ":gymnasium_robotics_envpool",
    ],
    imports = ["../.."],
    deps = [
        "//envpool/python:api",
    ],
)

py_library(
    name = "gymnasium_robotics_registration",
    srcs = ["registration.py"],
    imports = ["../.."],
    deps = [
        ":gymnasium_robotics",
        "//envpool:registration",
    ],
)

py_test(
    name = "gymnasium_robotics_test",
    size = "enormous",
    srcs = ["gymnasium_robotics_test.py"],
    imports = ["../.."],
    deps = [
        ":gymnasium_robotics",
        ":gymnasium_robotics_registration",
        "//envpool:registration",
        requirement("dm-env"),
        requirement("absl-py"),
        requirement("gymnasium"),
        requirement("gymnasium-robotics"),
        requirement("mujoco"),
        requirement("numpy"),
    ],
)
