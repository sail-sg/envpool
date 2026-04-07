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
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//envpool:requirements.bzl", "requirement")
load("//third_party:common.bzl", "copy_to_directory")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mujoco_render",
    srcs = ["offscreen_renderer.cc"],
    hdrs = ["offscreen_renderer.h"],
    linkopts = select({
        "//:linux": [
            "-lEGL",
            "-lOpenGL",
        ],
        "//:windows": [
            "gdi32.lib",
            "opengl32.lib",
            "user32.lib",
        ],
        "//conditions:default": [
            "-framework OpenGL",
        ],
    }),
    deps = ["@mujoco//:mujoco_lib"],
)

py_library(
    name = "mujoco",
    srcs = ["__init__.py"],
    imports = ["../.."],
    deps = [
        ":mujoco_dmc",
        ":mujoco_gym",
        ":robotics",
    ],
)

copy_to_directory(
    name = "gen_mujoco_gym_xml",
    srcs = [
        "//third_party/mujoco_gym_xml_patches:patches",
        "@mujoco_gym_xml",
    ],
    out = "assets_gym",
    strip_prefix = "assets/",
)

copy_to_directory(
    name = "gen_mujoco_dmc_xml",
    srcs = ["@mujoco_dmc_xml"],
    out = "assets_dmc",
    strip_prefix = "suite/",
)

copy_to_directory(
    name = "gen_gymnasium_robotics_assets",
    srcs = ["@gymnasium_robotics_assets"],
    out = "robotics/assets",
    strip_prefix = "gymnasium_robotics_assets/assets/",
)

cc_library(
    name = "mujoco_gym_env",
    hdrs = [
        "frame_stack.h",
        "gym/ant.h",
        "gym/half_cheetah.h",
        "gym/hopper.h",
        "gym/humanoid.h",
        "gym/humanoid_standup.h",
        "gym/inverted_double_pendulum.h",
        "gym/inverted_pendulum.h",
        "gym/mujoco_env.h",
        "gym/pusher.h",
        "gym/reacher.h",
        "gym/swimmer.h",
        "gym/walker2d.h",
        "offscreen_renderer.h",
    ],
    data = [
        ":gen_mujoco_gym_xml",
    ],
    deps = [
        ":mujoco_render",
        "//envpool/core:async_envpool",
        "@mujoco//:mujoco_lib",
    ],
)

pybind_extension(
    name = "mujoco_gym_envpool",
    srcs = [
        "gym/mujoco_envpool.cc",
    ],
    deps = [
        ":mujoco_gym_env",
        "//envpool/core:py_envpool",
    ],
)

cc_library(
    name = "mujoco_dmc_env",
    srcs = [
        "dmc/mujoco_env.cc",
        "dmc/utils.cc",
    ],
    hdrs = [
        "dmc/acrobot.h",
        "dmc/ball_in_cup.h",
        "dmc/cartpole.h",
        "dmc/cheetah.h",
        "dmc/dog.h",
        "dmc/finger.h",
        "dmc/fish.h",
        "dmc/hopper.h",
        "dmc/humanoid.h",
        "dmc/humanoid_CMU.h",
        "dmc/lqr.h",
        "dmc/manipulator.h",
        "dmc/mujoco_env.h",
        "dmc/pendulum.h",
        "dmc/point_mass.h",
        "dmc/quadruped.h",
        "dmc/reacher.h",
        "dmc/stacker.h",
        "dmc/swimmer.h",
        "dmc/utils.h",
        "dmc/walker.h",
        "frame_stack.h",
        "offscreen_renderer.h",
    ],
    data = [":gen_mujoco_dmc_xml"],
    deps = [
        ":mujoco_render",
        "//envpool/core:async_envpool",
        "@mujoco//:mujoco_lib",
        "@mujoco//:mujoco_stl_decoder_plugin_lib",
        "@pugixml",
    ],
)

pybind_extension(
    name = "mujoco_dmc_envpool",
    srcs = [
        "dmc/mujoco_envpool.cc",
    ],
    deps = [
        ":mujoco_dmc_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "mujoco_dmc",
    srcs = ["dmc/__init__.py"],
    data = [
        ":gen_mujoco_dmc_xml",
        ":mujoco_dmc_envpool",
    ],
    imports = ["../.."],
    deps = ["//envpool/python:api"],
)

py_library(
    name = "mujoco_gym",
    srcs = ["gym/__init__.py"],
    data = [
        ":gen_mujoco_gym_xml",
        ":mujoco_gym_envpool",
    ],
    imports = ["../.."],
    deps = ["//envpool/python:api"],
)

cc_library(
    name = "robotics_env",
    srcs = ["robotics/utils.cc"],
    hdrs = [
        "frame_stack.h",
        "robotics/adroit.h",
        "robotics/fetch.h",
        "robotics/hand.h",
        "robotics/kitchen.h",
        "robotics/mujoco_env.h",
        "robotics/point_maze.h",
        "robotics/utils.h",
    ],
    data = [":gen_gymnasium_robotics_assets"],
    deps = [
        ":mujoco_render",
        "//envpool/core:async_envpool",
        "@mujoco//:mujoco_obj_decoder_plugin_lib",
        "@mujoco//:mujoco_stl_decoder_plugin_lib",
    ],
)

pybind_extension(
    name = "robotics_envpool",
    srcs = [
        "robotics/mujoco_plugin_init.cc",
        "robotics/robotics_envpool.cc",
    ],
    deps = [
        ":robotics_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "robotics",
    srcs = ["robotics/__init__.py"],
    data = [
        ":gen_gymnasium_robotics_assets",
        ":robotics_envpool",
    ],
    imports = ["../.."],
    deps = ["//envpool/python:api"],
)

cc_test(
    name = "mujoco_envpool_test",
    size = "enormous",
    srcs = ["gym/mujoco_gym_envpool_test.cc"],
    deps = [
        ":mujoco_gym_env",
        "//envpool/core:logging",
        "@com_google_googletest//:gtest_main",
    ],
)

py_library(
    name = "mujoco_dmc_registration",
    srcs = ["dmc/registration.py"],
    imports = ["../.."],
    deps = ["//envpool:registration"],
)

py_library(
    name = "mujoco_gym_registration",
    srcs = ["gym/registration.py"],
    imports = ["../.."],
    deps = ["//envpool:registration"],
)

py_library(
    name = "robotics_registration",
    srcs = ["robotics/registration.py"],
    imports = ["../.."],
    deps = [
        ":robotics",
        "//envpool:registration",
    ],
)

py_test(
    name = "mujoco_render_test",
    size = "enormous",
    srcs = ["gym/mujoco_render_test.py"],
    imports = ["../.."],
    deps = [
        ":mujoco_gym",
        ":mujoco_gym_registration",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("gymnasium"),
    ],
)

py_test(
    name = "mujoco_dmc_render_test",
    size = "enormous",
    srcs = ["dmc/mujoco_dmc_render_test.py"],
    imports = ["../.."],
    deps = [
        ":mujoco_dmc",
        ":mujoco_dmc_registration",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("dm-control"),
        requirement("gymnasium"),
    ],
)

py_test(
    name = "mujoco_dmc_pixel_observation_test",
    size = "enormous",
    srcs = ["dmc/mujoco_dmc_pixel_observation_test.py"],
    imports = ["../.."],
    deps = [
        ":mujoco_dmc",
        ":mujoco_dmc_registration",
        ":mujoco_pixel_observation_test_utils",
        requirement("dm-control"),
    ],
)

py_test(
    name = "mujoco_gym_pixel_observation_test",
    size = "enormous",
    srcs = ["gym/mujoco_gym_pixel_observation_test.py"],
    imports = ["../.."],
    deps = [
        ":mujoco_gym",
        ":mujoco_gym_registration",
        ":mujoco_pixel_observation_test_utils",
    ],
)

py_test(
    name = "robotics_pixel_observation_test",
    size = "enormous",
    srcs = ["robotics/robotics_pixel_observation_test.py"],
    imports = ["../.."],
    deps = [
        ":mujoco_pixel_observation_test_utils",
        ":robotics",
        ":robotics_registration",
        requirement("gymnasium-robotics"),
    ],
)

py_library(
    name = "mujoco_pixel_observation_test_utils",
    srcs = ["pixel_observation_test_utils.py"],
    imports = ["../.."],
    deps = [
        "//envpool:registration",
        "//envpool/python:glfw_context",
        requirement("absl-py"),
        requirement("gymnasium"),
        requirement("numpy"),
    ],
)

py_test(
    name = "mujoco_plugin_registration_test",
    srcs = ["mujoco_plugin_registration_test.py"],
    imports = ["../.."],
    deps = [
        ":mujoco_dmc",
        ":robotics",
        requirement("absl-py"),
    ],
)

test_suite(
    name = "mujoco_pixel_observation_test",
    tests = [
        ":mujoco_dmc_pixel_observation_test",
        ":mujoco_gym_pixel_observation_test",
        ":robotics_pixel_observation_test",
    ],
)

py_test(
    name = "mujoco_gym_deterministic_test",
    size = "enormous",
    srcs = ["gym/mujoco_gym_deterministic_test.py"],
    imports = ["../.."],
    deps = [
        ":mujoco_gym",
        ":mujoco_gym_registration",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("gymnasium"),
    ],
)

py_test(
    name = "mujoco_gym_align_test",
    size = "enormous",
    srcs = ["gym/mujoco_gym_align_test.py"],
    imports = ["../.."],
    deps = [
        ":mujoco_gym",
        ":mujoco_gym_registration",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("gymnasium"),
        requirement("imageio"),
        requirement("mujoco"),
    ],
)

py_test(
    name = "mujoco_dmc_suite_deterministic_test",
    size = "enormous",
    srcs = ["dmc/mujoco_dmc_suite_deterministic_test.py"],
    imports = ["../.."],
    deps = [
        ":mujoco_dmc",
        ":mujoco_dmc_registration",
        requirement("numpy"),
        requirement("absl-py"),
    ],
)

py_test(
    name = "mujoco_dmc_suite_align_test",
    size = "enormous",
    srcs = ["dmc/mujoco_dmc_suite_align_test.py"],
    imports = ["../.."],
    deps = [
        ":mujoco_dmc",
        ":mujoco_dmc_registration",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("mujoco"),
        requirement("dm-env"),
        requirement("dm-control"),
    ],
)

py_test(
    name = "mujoco_dmc_suite_ext_deterministic_test",
    size = "enormous",
    srcs = ["dmc/mujoco_dmc_suite_ext_deterministic_test.py"],
    imports = ["../.."],
    deps = [
        ":mujoco_dmc",
        ":mujoco_dmc_registration",
        requirement("numpy"),
        requirement("absl-py"),
    ],
)

py_test(
    name = "mujoco_dmc_suite_ext_align_test",
    size = "enormous",
    srcs = ["dmc/mujoco_dmc_suite_ext_align_test.py"],
    imports = ["../.."],
    deps = [
        ":mujoco_dmc",
        ":mujoco_dmc_registration",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("mujoco"),
        requirement("dm-env"),
        requirement("dm-control"),
    ],
)

py_test(
    name = "robotics_test",
    size = "enormous",
    srcs = ["robotics/robotics_test.py"],
    imports = ["../.."],
    deps = [
        ":robotics",
        ":robotics_registration",
        requirement("dm-env"),
        requirement("absl-py"),
        requirement("gymnasium"),
        requirement("gymnasium-robotics"),
        requirement("mujoco"),
        requirement("numpy"),
    ],
)
