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

package(default_visibility = ["//visibility:public"])

exports_files([
    "workspace0.bzl",
    "workspace1.bzl",
])

py_library(
    name = "registration",
    srcs = ["registration.py"],
)

py_library(
    name = "entry",
    srcs = ["entry.py"],
    deps = [
        "//envpool/atari:atari_registration",
        "//envpool/box2d:box2d_registration",
        "//envpool/classic_control:classic_control_registration",
        "//envpool/mujoco:mujoco_dmc_registration",
        "//envpool/mujoco:mujoco_gym_registration",
        "//envpool/procgen:procgen_registration",
        "//envpool/toy_text:toy_text_registration",
        "//envpool/vizdoom:vizdoom_registration",
        "//envpool/football:football_registration",
    ],
)

py_library(
    name = "envpool",
    srcs = ["__init__.py"],
    deps = [
        ":entry",
        ":registration",
        "//envpool/atari",
        "//envpool/box2d",
        "//envpool/classic_control",
        "//envpool/football",
        "//envpool/mujoco:mujoco_dmc",
        "//envpool/mujoco:mujoco_gym",
        "//envpool/procgen",
        "//envpool/python",
        "//envpool/toy_text",
        "//envpool/vizdoom",
    ],
)

py_test(
    name = "make_test",
    srcs = ["make_test.py"],
    deps = [
        ":envpool",
        requirement("absl-py"),
    ],
)
