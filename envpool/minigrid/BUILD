# Copyright 2023 Garena Online Private Limited
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
    name = "minigrid_env",
    srcs = [
        "impl/minigrid_env.cc",
    ],
    hdrs = [
        "impl/minigrid_env.h",
        "impl/utils.h",
        "minigrid.h",
    ],
    deps = [
        "//envpool/core:async_envpool",
        "//envpool/core:logging",
        "//envpool/core:py_envpool",
    ],
)

pybind_extension(
    name = "minigrid_envpool",
    srcs = ["minigrid.cc"],
    deps = [
        ":minigrid_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "minigrid",
    srcs = ["__init__.py"],
    data = [":minigrid_envpool.so"],
    deps = [
        "//envpool/python:api",
        requirement("numpy"),
    ],
)

py_library(
    name = "minigrid_registration",
    srcs = ["registration.py"],
    deps = [
        "//envpool:registration",
    ],
)

py_test(
    name = "minigrid_align_test",
    size = "enormous",
    srcs = ["minigrid_align_test.py"],
    deps = [
        ":minigrid",
        ":minigrid_registration",
        requirement("absl-py"),
        requirement("gym"),
        requirement("numpy"),
        requirement("minigrid"),
    ],
)

py_test(
    name = "minigrid_deterministic_test",
    size = "enormous",
    srcs = ["minigrid_deterministic_test.py"],
    deps = [
        ":minigrid",
        ":minigrid_registration",
        requirement("absl-py"),
        requirement("gym"),
        requirement("numpy"),
    ],
)
