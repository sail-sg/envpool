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

_COPTS = select({
    "//:windows": ["/fp:strict"],
    "//conditions:default": ["-ffp-contract=off"],
})

cc_library(
    name = "model",
    srcs = [
        "model.cc",
        "render.cc",
    ],
    hdrs = [
        "model.h",
    ],
    copts = _COPTS,
    deps = [
        "//envpool/utils:random",
        "//third_party/mpe2:metadata_headers",
        "@freetype",
    ],
)

cc_library(
    name = "mpe2_env",
    hdrs = ["mpe2.h"],
    copts = _COPTS,
    deps = [
        ":model",
        "//envpool/core:async_envpool",
        "//envpool/core:logging",
    ],
)

pybind_extension(
    name = "mpe2_envpool",
    srcs = ["mpe2.cc"],
    copts = _COPTS,
    deps = [
        ":mpe2_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "mpe2",
    srcs = ["__init__.py"],
    data = [
        "FONT_LICENSE",
        "LICENSE",
        ":mpe2_envpool",
    ],
    imports = ["../.."],
    deps = ["//envpool/python:api"],
)

py_library(
    name = "registration",
    srcs = ["registration.py"],
    data = [":mpe2_envpool"],
    imports = ["../.."],
    deps = ["//envpool:registration"],
)

py_library(
    name = "oracle",
    testonly = True,
    srcs = ["oracle.py"],
    deps = [
        ":mpe2",
        "@mpe2_upstream//:oracle",
        requirement("numpy"),
    ],
)

py_test(
    name = "mpe2_test",
    size = "large",
    srcs = ["mpe2_test.py"],
    imports = ["../.."],
    deps = [
        ":mpe2",
        ":oracle",
        ":registration",
        "//envpool/python:seed_test_utils",
        "@mpe2_upstream//:oracle",
        requirement("absl-py"),
        requirement("numpy"),
        requirement("scipy"),
    ],
)

py_binary(
    name = "generate_docs",
    testonly = True,
    srcs = ["generate_docs.py"],
    deps = [
        ":oracle",
        ":registration",
        requirement("numpy"),
        requirement("pillow"),
    ],
)
