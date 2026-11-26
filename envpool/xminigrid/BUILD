# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("//envpool:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "xminigrid_env",
    srcs = [
        "layout.cc",
        "render.cc",
        "xminigrid.cc",
    ],
    hdrs = ["xminigrid.h"],
    textual_hdrs = ["//third_party/xminigrid:tasks.inc"],
    deps = [
        "//envpool/core:env",
        "//envpool/minigrid:tile_render",
    ],
)

pybind_extension(
    name = "xminigrid_envpool",
    srcs = ["bindings.cc"],
    deps = [
        ":xminigrid_env",
        "//envpool/core:async_envpool",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "xminigrid",
    srcs = ["__init__.py"],
    data = [":xminigrid_envpool"],
    imports = ["../.."],
    deps = ["//envpool/python:api"],
)

py_library(
    name = "registration",
    srcs = ["registration.py"],
    data = [":xminigrid_envpool"],
    imports = ["../.."],
    deps = ["//envpool:registration"],
)

py_library(
    name = "oracle",
    testonly = True,
    srcs = ["oracle.py"],
    deps = [
        ":xminigrid",
        "@xminigrid_upstream//:oracle",
    ],
)

py_test(
    name = "xminigrid_test",
    size = "large",
    srcs = ["xminigrid_test.py"],
    deps = [
        ":oracle",
        ":registration",
        requirement("absl-py"),
    ],
)

py_binary(
    name = "generate_docs",
    testonly = True,
    srcs = ["generate_docs.py"],
    deps = [
        ":oracle",
        ":registration",
        requirement("pillow"),
    ],
)
