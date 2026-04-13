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

load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//envpool:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

exports_files([
    "workspace0.bzl",
    "workspace1.bzl",
])

py_library(
    name = "registration",
    srcs = ["registration.py"],
    imports = [".."],
)

py_library(
    name = "entry",
    srcs = ["entry.py"],
    imports = [".."],
    deps = [
        "//envpool/atari:atari_registration",
        "//envpool/box2d:box2d_registration",
        "//envpool/classic_control:classic_control_registration",
        "//envpool/gfootball:gfootball_registration",
        "//envpool/highway:highway_registration",
        "//envpool/minigrid:minigrid_registration",
        "//envpool/mujoco:metaworld_registration",
        "//envpool/mujoco:mujoco_dmc_registration",
        "//envpool/mujoco:mujoco_gym_registration",
        "//envpool/mujoco:robotics_registration",
        "//envpool/procgen:procgen_registration",
        "//envpool/toy_text:toy_text_registration",
        "//envpool/vizdoom:vizdoom_registration",
    ],
)

py_library(
    name = "envpool",
    srcs = ["__init__.py"],
    data = ["py.typed"],
    imports = [".."],
    deps = [
        ":entry",
        ":registration",
        "//envpool/atari",
        "//envpool/box2d",
        "//envpool/classic_control",
        "//envpool/gfootball",
        "//envpool/highway",
        "//envpool/minigrid",
        "//envpool/mujoco",
        "//envpool/procgen",
        "//envpool/python",
        "//envpool/toy_text",
        "//envpool/vizdoom",
    ],
)

py_test(
    name = "make_test",
    timeout = "long",
    srcs = ["make_test.py"],
    imports = [".."],
    deps = [
        ":envpool",
        requirement("absl-py"),
    ],
)
