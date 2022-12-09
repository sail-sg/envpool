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

package(default_visibility = ["//visibility:public"])

genrule(
    name = "gen_pretrain_weight",
    srcs = [
        "@pretrain_weight//:pth",
    ],
    outs = [
        "policy-d1.pth",
        "policy-d3.pth",
    ],
    cmd = "cp $(SRCS) $(@D)",
)

genrule(
    name = "gen_vizdoom_maps",
    srcs = [
        "@vizdoom_lib//:vizdoom_maps",
        "@vizdoom_extra_maps//:vizdoom_maps",
    ],
    outs = ["maps"],
    cmd = "mkdir -p $(OUTS) && cp $(SRCS) $(OUTS)",
)

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
    deps = ["@vizdoom_lib"],
)

cc_library(
    name = "vizdoom_env",
    hdrs = ["vizdoom_env.h"],
    deps = [
        ":utils",
        "//envpool/core:async_envpool",
        "//envpool/utils:image_process",
    ],
)

pybind_extension(
    name = "vizdoom_envpool",
    srcs = ["vizdoom_envpool.cc"],
    linkopts = [
        "-ldl",
    ],
    deps = [
        ":vizdoom_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "vizdoom",
    srcs = ["__init__.py"],
    data = [
        ":gen_vizdoom_maps",
        ":vizdoom_envpool.so",
        "//envpool/vizdoom/bin:freedoom",
        "//envpool/vizdoom/bin:vizdoom_bin",
        "//envpool/vizdoom/bin:vizdoom_pk3",
    ],
    deps = ["//envpool/python:api"],
)

py_library(
    name = "vizdoom_registration",
    srcs = ["registration.py"],
    deps = ["//envpool:registration"],
)

py_test(
    name = "vizdoom_test",
    srcs = ["vizdoom_test.py"],
    deps = [
        ":vizdoom",
        ":vizdoom_registration",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("opencv-python-headless"),
    ],
)

py_test(
    name = "vizdoom_pretrain_test",
    size = "enormous",
    srcs = ["vizdoom_pretrain_test.py"],
    data = [":gen_pretrain_weight"],
    deps = [
        ":vizdoom",
        ":vizdoom_registration",
        "//envpool/atari:atari_network",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("tianshou"),
        requirement("torch"),
    ],
)
