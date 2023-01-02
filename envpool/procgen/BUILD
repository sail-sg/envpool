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

load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

genrule(
    name = "gen_procgen_assets",
    srcs = ["@procgen//:procgen_assets"],
    outs = ["assets"],
    cmd = "mkdir -p $(OUTS) && cp -r $(SRCS) $(OUTS)",
)

cc_library(
    name = "procgen_env",
    hdrs = ["procgen_env.h"],
    data = [
        ":gen_procgen_assets",
    ],
    deps = [
        "//envpool/core:async_envpool",
        "@procgen//:procgen",
    ],
)

cc_test(
    name = "procgen_env_test",
    srcs = ["procgen_env_test.cc"],
    deps = [
        ":procgen_env",
        "@com_google_googletest//:gtest_main",
    ],
)

pybind_extension(
    name = "procgen_envpool",
    srcs = ["procgen_envpool.cc"],
    linkopts = [
        "-ldl",
    ],
    deps = [
        ":procgen_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "procgen",
    srcs = ["__init__.py"],
    data = [":procgen_envpool.so"],
    deps = ["//envpool/python:api"],
)

py_library(
    name = "procgen_registration",
    srcs = ["registration.py"],
    deps = [
        "//envpool:registration",
    ],
)

py_test(
    name = "procgen_test",
    srcs = ["procgen_test.py"],
    deps = [
        ":procgen",
        ":procgen_registration",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("gym"),
    ],
)
