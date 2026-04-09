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

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "highway_env",
    srcs = ["highway_env.cc"],
    hdrs = ["highway_env.h"],
    deps = [
        "//envpool/core:async_envpool",
    ],
)

cc_library(
    name = "official_lane",
    srcs = ["official_lane.cc"],
    hdrs = ["official_lane.h"],
)

cc_library(
    name = "official_road",
    srcs = ["official_road.cc"],
    hdrs = ["official_road.h"],
    deps = [
        ":official_lane",
    ],
)

cc_library(
    name = "official_vehicle",
    srcs = ["official_vehicle.cc"],
    hdrs = ["official_vehicle.h"],
    deps = [
        ":official_lane",
        ":official_road",
    ],
)

cc_library(
    name = "official_scene",
    srcs = ["official_scene.cc"],
    hdrs = ["official_scene.h"],
    deps = [
        ":official_lane",
        ":official_road",
        ":official_vehicle",
    ],
)

cc_library(
    name = "official_observation",
    srcs = ["official_observation.cc"],
    hdrs = ["official_observation.h"],
    deps = [
        ":official_scene",
    ],
)

cc_library(
    name = "official_task",
    srcs = ["official_task.cc"],
    hdrs = ["official_task.h"],
    deps = [
        ":official_lane",
        ":official_scene",
    ],
)

pybind_extension(
    name = "highway_envpool",
    srcs = [
        "highway_envpool.cc",
        "native_task_env.h",
    ],
    deps = [
        ":highway_env",
        ":official_observation",
        ":official_task",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "highway",
    srcs = ["__init__.py"],
    data = [":highway_envpool"],
    imports = ["../.."],
    deps = [
        "//envpool/python:api",
        requirement("numpy"),
    ],
)

py_library(
    name = "highway_registration",
    srcs = ["registration.py"],
    imports = ["../.."],
    deps = [
        "//envpool:registration",
    ],
)

py_test(
    name = "highway_deterministic_test",
    size = "enormous",
    srcs = ["highway_deterministic_test.py"],
    imports = ["../.."],
    deps = [
        ":highway",
        ":highway_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "highway_align_test",
    size = "enormous",
    srcs = [
        "highway_align_test.py",
        "highway_oracle_util.py",
    ],
    imports = ["../.."],
    deps = [
        ":highway",
        ":highway_registration",
        requirement("absl-py"),
        requirement("gymnasium"),
        requirement("highway-env"),
        requirement("numpy"),
    ],
)

py_test(
    name = "highway_render_test",
    size = "enormous",
    srcs = [
        "highway_align_test.py",
        "highway_oracle_util.py",
        "highway_render_test.py",
    ],
    imports = ["../.."],
    deps = [
        ":highway",
        ":highway_registration",
        requirement("absl-py"),
        requirement("gymnasium"),
        requirement("highway-env"),
        requirement("numpy"),
    ],
)

py_test(
    name = "highway_official_align_test",
    size = "enormous",
    srcs = [
        "highway_align_test.py",
        "highway_official_align_test.py",
        "highway_official_coverage_test.py",
        "highway_oracle_util.py",
    ],
    imports = ["../.."],
    deps = [
        ":highway",
        ":highway_registration",
        requirement("absl-py"),
        requirement("gymnasium"),
        requirement("highway-env"),
        requirement("numpy"),
    ],
)

py_test(
    name = "highway_official_coverage_test",
    size = "enormous",
    srcs = [
        "highway_official_coverage_test.py",
        "highway_oracle_util.py",
    ],
    imports = ["../.."],
    deps = [
        ":highway",
        ":highway_registration",
        requirement("absl-py"),
        requirement("gymnasium"),
        requirement("highway-env"),
        requirement("numpy"),
    ],
)
