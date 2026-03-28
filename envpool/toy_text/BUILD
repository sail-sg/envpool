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

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//envpool:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "toy_text_env",
    hdrs = [
        "blackjack.h",
        "catch.h",
        "cliffwalking.h",
        "frozen_lake.h",
        "nchain.h",
        "taxi.h",
    ],
    deps = [
        "//envpool/core:async_envpool",
    ],
)

pybind_extension(
    name = "toy_text_envpool",
    srcs = [
        "toy_text.cc",
    ],
    deps = [
        ":toy_text_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "toy_text",
    srcs = ["__init__.py"],
    imports = ["../.."],
    data = [":toy_text_envpool"],
    deps = ["//envpool/python:api"],
)

py_library(
    name = "toy_text_registration",
    srcs = ["registration.py"],
    imports = ["../.."],
    deps = [
        "//envpool:registration",
    ],
)

py_test(
    name = "toy_text_test",
    srcs = ["toy_text_test.py"],
    imports = ["../.."],
    deps = [
        ":toy_text",
        ":toy_text_registration",
        requirement("absl-py"),
        requirement("dm_env"),
        requirement("gymnasium"),
        requirement("numpy"),
        requirement("scipy"),
        requirement("pygame"),
    ],
)
