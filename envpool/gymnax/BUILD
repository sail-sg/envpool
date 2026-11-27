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

load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("//envpool:requirements.bzl", "requirement")
load("//third_party/gymnax:repo.bzl", "MNIST_FILES")

package(default_visibility = ["//visibility:public"])

copy_file(
    name = "license",
    src = "@gymnax_upstream//:LICENSE",
    out = "LICENSE",
)

[copy_file(
    name = "copy_" + f,
    testonly = f.startswith("t10k"),
    src = "@mnist//:" + f,
    out = f,
) for f in MNIST_FILES]

cc_library(
    name = "games",
    srcs = [
        "arcade.cc",
        "games.cc",
        "robot.cc",
        "seaquest.cc",
    ],
    hdrs = ["games.h"],
    textual_hdrs = ["//third_party/gymnax:tasks.inc"],
    deps = [
        "//envpool/utils:grid_render",
        "@opencv",
        "@zlib",
    ],
)

pybind_extension(
    name = "gymnax_envpool",
    srcs = ["bindings.cc"],
    deps = [
        ":games",
        "//envpool/core:async_envpool",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "gymnax",
    srcs = ["__init__.py"],
    data = [
        "NOTICE",
        ":gymnax_envpool",
        ":license",
    ] + [f for f in MNIST_FILES if f.startswith("train")],
    imports = ["../.."],
    deps = ["//envpool/python:api"],
)

py_library(
    name = "registration",
    srcs = ["registration.py"],
    imports = ["../.."],
    deps = [
        ":gymnax",
        "//envpool:registration",
        "//envpool/classic_control",
        "//envpool/toy_text",
    ],
)

py_library(
    name = "oracle",
    testonly = True,
    srcs = ["oracle.py"],
    data = MNIST_FILES.keys(),
    deps = ["@gymnax_upstream//:oracle"],
)

py_test(
    name = "gymnax_test",
    size = "large",
    srcs = ["gymnax_test.py"],
    env = {"MPLBACKEND": "Agg"},
    deps = [
        ":oracle",
        ":registration",
        "//envpool/classic_control:classic_control_registration",
        requirement("absl-py"),
    ],
)

py_test(
    name = "sampling_test",
    size = "large",
    srcs = ["sampling_test.py"],
    deps = [
        ":oracle",
        ":registration",
        "//envpool/python:seed_test_utils",
        requirement("absl-py"),
    ],
)

py_binary(
    name = "generate_docs",
    testonly = True,
    srcs = ["generate_docs.py"],
    env = {"MPLBACKEND": "Agg"},
    deps = [
        ":oracle",
        ":registration",
        requirement("pillow"),
    ],
)
