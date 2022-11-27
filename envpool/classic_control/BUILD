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

cc_library(
    name = "classic_control_env",
    hdrs = [
        "acrobot.h",
        "cartpole.h",
        "mountain_car.h",
        "mountain_car_continuous.h",
        "pendulum.h",
    ],
    deps = [
        "//envpool/core:async_envpool",
    ],
)

pybind_extension(
    name = "classic_control_envpool",
    srcs = ["classic_control.cc"],
    deps = [
        ":classic_control_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "classic_control",
    srcs = ["__init__.py"],
    data = [":classic_control_envpool.so"],
    deps = ["//envpool/python:api"],
)

py_library(
    name = "classic_control_registration",
    srcs = ["registration.py"],
    deps = [
        "//envpool:registration",
    ],
)

py_test(
    name = "classic_control_test",
    size = "enormous",
    srcs = ["classic_control_test.py"],
    deps = [
        ":classic_control",
        ":classic_control_registration",
        requirement("absl-py"),
        requirement("dm_env"),
        requirement("gym"),
        requirement("numpy"),
        requirement("pygame"),
    ],
)
