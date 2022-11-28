# Copyright 2022 Garena Online Private Limited
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

load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

py_library(
    name = "dummy",
    srcs = ["__init__.py"],
    data = [":dummy_envpool.so"],
    deps = ["//envpool/python:api"],
)

cc_library(
    name = "dummy_envpool_h",
    hdrs = ["dummy_envpool.h"],
    deps = [
        "//envpool/core:async_envpool",
        "//envpool/core:env",
        "//envpool/core:env_spec",
    ],
)

cc_test(
    name = "dummy_envpool_test",
    size = "enormous",
    srcs = ["dummy_envpool_test.cc"],
    deps = [
        ":dummy_envpool_h",
        "@com_google_googletest//:gtest_main",
    ],
)

py_test(
    name = "dummy_py_envpool_test",
    srcs = ["dummy_py_envpool_test.py"],
    data = ["dummy_envpool.so"],
    deps = [
        requirement("numpy"),
        requirement("absl-py"),
    ],
)

pybind_extension(
    name = "dummy_envpool",
    srcs = [
        "dummy_envpool.cc",
    ],
    linkopts = [
        "-ldl",
    ],
    deps = [
        ":dummy_envpool_h",
        "//envpool/core:py_envpool",
    ],
)
