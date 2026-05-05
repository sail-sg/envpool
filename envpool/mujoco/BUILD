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
load("@python_versions//3.12:defs.bzl", py_binary_312 = "py_binary")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//envpool:requirements.bzl", "requirement")
load("//third_party:common.bzl", "copy_to_directory")
load("//third_party/metaworld_assets:defs.bzl", "metaworld_runtime_assets")
load("//third_party/myosuite:defs.bzl", "myosuite_runtime_assets")
load("//third_party/myosuite:oracle_requirements.bzl", "oracle_requirement")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "release_assets",
    values = {"define": "envpool_release_assets=true"},
)

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
        ":metaworld",
        ":mujoco_dmc",
        ":mujoco_gym",
        ":myosuite",
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

metaworld_runtime_assets(
    name = "gen_metaworld_assets",
    srcs = ["@metaworld_assets"],
    out = "metaworld/assets",
    compact = select({
        ":release_assets": True,
        "//conditions:default": False,
    }),
    strip_prefix = "metaworld_assets/metaworld/assets/",
)

myosuite_runtime_assets(
    name = "gen_myosuite_assets",
    srcs = [
        "@myosuite_furniture_sim//:source",
        "@myosuite_mpl_sim//:source",
        "@myosuite_myo_sim//:source",
        "@myosuite_object_sim//:source",
        "@myosuite_source//:runtime_assets",
        "@myosuite_ycb_sim//:source",
    ],
    out = "myosuite/assets",
    metadata_srcs = ["//third_party/myosuite:myosuite_generated_json"],
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
        "dmc/mujoco_plugin_init.cc",
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
        ":metaworld_env",
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

cc_library(
    name = "metaworld_env",
    hdrs = [
        "metaworld/metaworld_env.h",
        "metaworld/tasks.h",
    ],
    data = [":gen_metaworld_assets"],
    deps = [
        ":robotics_env",
        "//envpool/core:async_envpool",
    ],
)

py_library(
    name = "metaworld",
    srcs = ["metaworld/__init__.py"],
    data = [
        ":gen_metaworld_assets",
        ":robotics_envpool",
    ],
    imports = ["../.."],
    deps = ["//envpool/python:api"],
)

cc_library(
    name = "myosuite_env",
    hdrs = [
        "frame_stack.h",
        "myosuite/myosuite_env.h",
        "offscreen_renderer.h",
    ],
    data = [":gen_myosuite_assets"],
    deps = [
        ":mujoco_render",
        ":robotics_env",
        "//envpool/core:async_envpool",
        "//third_party/myosuite:myosuite_tasks",
        "@mujoco//:mujoco_lib",
        "@mujoco//:mujoco_obj_decoder_plugin_lib",
        "@mujoco//:mujoco_stl_decoder_plugin_lib",
    ],
)

cc_library(
    name = "myosuite_clang_tidy",
    srcs = ["myosuite/myosuite_clang_tidy.cc"],
    deps = [":myosuite_env"],
)

cc_library(
    name = "myosuite_envpool_module",
    srcs = [
        "myosuite/mujoco_plugin_init.cc",
        "myosuite/myosuite_envpool.cc",
    ],
    # The native MyoSuite runtime is header-only under myosuite_env.h and is
    # compiled into this module object. Coverage instrumentation changes Linux
    # floating-point codegen enough to make long oracle rollouts diverge; the
    # pybind_extension wrapper still links under coverage so instrumented
    # transitive deps resolve their coverage runtime symbols.
    copts = select({
        "//:windows": [
            # CI fastbuild otherwise compiles the header-only MyoSuite runtime
            # at /Od while the pinned official oracle wheel is release-built.
            "/O2",
        ],
        "//:linux_x86_64": [
            # CI fastbuild otherwise compiles the header-only MyoSuite runtime
            # at -O0 while the pinned official oracle wheel is release-built.
            "-O3",
            "-fno-profile-arcs",
            "-fno-test-coverage",
        ],
        "//:linux": [
            "-fno-profile-arcs",
            "-fno-test-coverage",
        ],
        "@platforms//os:osx": [
            "-fno-coverage-mapping",
            "-fno-profile-instr-generate",
        ],
        "//conditions:default": [],
    }),
    features = ["-coverage"],
    deps = [
        ":myosuite_env",
        "//envpool/core:py_envpool",
    ],
    alwayslink = True,
)

pybind_extension(
    name = "myosuite_envpool",
    deps = [
        ":myosuite_envpool_module",
    ],
)

py_binary_312(
    name = "generate_myosuite_render_sample",
    srcs = [
        "myosuite/myosuite_oracle_probe.py",
        "//third_party/myosuite:generate_render_sample.py",
    ],
    data = select({
        # Linux oracle rendering uses the pinned pip MuJoCo wheel directly.
        "//:linux": [],
        "//conditions:default": ["@mujoco//:mujoco_shared_lib"],
    }) + [
        ":myosuite_oracle_probe",
        "@myosuite_furniture_sim//:source",
        "@myosuite_mpl_sim//:source",
        "@myosuite_myo_sim//:source",
        "@myosuite_object_sim//:source",
        "@myosuite_source//:source",
        "@myosuite_ycb_sim//:source",
    ],
    imports = ["../.."],
    main = "generate_render_sample.py",
    deps = [
        ":myosuite",
        ":myosuite_registration",
        oracle_requirement("click"),
        oracle_requirement("dm-control"),
        oracle_requirement("flatten-dict"),
        oracle_requirement("gitpython"),
        oracle_requirement("gymnasium"),
        oracle_requirement("h5py"),
        oracle_requirement("mujoco"),
        oracle_requirement("numpy"),
        oracle_requirement("packaging"),
        oracle_requirement("pillow"),
        oracle_requirement("pink-noise-rl"),
        oracle_requirement("sk-video"),
        oracle_requirement("termcolor"),
    ],
)

py_library(
    name = "myosuite_task_metadata",
    srcs = ["myosuite/tasks.py"],
    data = [":gen_myosuite_assets"],
    imports = ["../.."],
)

py_library(
    name = "myosuite",
    srcs = ["myosuite/__init__.py"],
    data = [
        ":gen_myosuite_assets",
        ":myosuite_envpool",
    ],
    imports = ["../.."],
    deps = [
        ":myosuite_task_metadata",
        "//envpool/python:api",
        "//envpool/python:glfw_context",
    ],
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

py_library(
    name = "metaworld_registration",
    srcs = ["metaworld/registration.py"],
    imports = ["../.."],
    deps = [
        ":metaworld",
        "//envpool:registration",
    ],
)

py_library(
    name = "myosuite_registration",
    srcs = ["myosuite/registration.py"],
    imports = ["../.."],
    deps = [
        ":myosuite",
        ":myosuite_task_metadata",
        "//envpool:registration",
    ],
)

py_test(
    name = "metaworld_test",
    size = "enormous",
    srcs = ["metaworld/metaworld_test.py"],
    imports = ["../.."],
    deps = [
        ":metaworld",
        ":metaworld_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "myosuite_test",
    size = "enormous",
    srcs = ["myosuite/myosuite_test.py"],
    imports = ["../.."],
    shard_count = 8,
    tags = ["exclusive"],
    deps = [
        ":myosuite",
        ":myosuite_registration",
        ":myosuite_task_metadata",
        "//envpool/python:glfw_context",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "myosuite_render_test",
    size = "enormous",
    srcs = ["myosuite/myosuite_render_test.py"],
    data = [":myosuite_oracle_probe"],
    imports = ["../.."],
    shard_count = 8,
    tags = ["exclusive"],
    deps = [
        ":myosuite",
        ":myosuite_registration",
        ":myosuite_task_metadata",
        "//envpool/python:glfw_context",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

MYOSUITE_CGL_RENDER_REGRESSION_IDS = ",".join([
    "motorFingerReachFixed-v0",
    "motorFingerReachRandom-v0",
    "myoElbowPose1D6MExoFixed-v0",
    "myoElbowPose1D6MExoRandom-v0",
    "myoElbowPose1D6MFixed-v0",
    "myoElbowPose1D6MRandom-v0",
    "myoChallengeTableTennisP0-v0",
    "myoChallengeTableTennisP1-v0",
    "myoChallengeTableTennisP2-v0",
    "myoFatiElbowPose1D6MExoFixed-v0",
    "myoFatiElbowPose1D6MExoRandom-v0",
    "myoFatiElbowPose1D6MFixed-v0",
    "myoFatiElbowPose1D6MRandom-v0",
    "myoFingerPoseFixed-v0",
    "myoFingerPoseRandom-v0",
    "myoFingerReachFixed-v0",
    "myoFingerReachRandom-v0",
    "myoSarcElbowPose1D6MExoFixed-v0",
    "myoSarcElbowPose1D6MExoRandom-v0",
    "myoSarcElbowPose1D6MFixed-v0",
    "myoSarcElbowPose1D6MRandom-v0",
])

py_test(
    name = "myo_cgl_render_test",
    size = "enormous",
    srcs = ["myosuite/myosuite_render_test.py"],
    data = [":myosuite_oracle_probe"],
    env = {
        "MYOSUITE_RENDER_TASK_IDS": MYOSUITE_CGL_RENDER_REGRESSION_IDS,
    },
    imports = ["../.."],
    main = "myosuite/myosuite_render_test.py",
    tags = ["exclusive"],
    deps = [
        ":myosuite",
        ":myosuite_registration",
        ":myosuite_task_metadata",
        "//envpool/python:glfw_context",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "myosuite_oracle_align_test",
    size = "enormous",
    srcs = ["myosuite/myosuite_oracle_align_test.py"],
    data = [":myosuite_oracle_probe"],
    imports = ["../.."],
    shard_count = 4,
    tags = ["exclusive"],
    deps = [
        ":myosuite",
        ":myosuite_registration",
        ":myosuite_task_metadata",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_binary_312(
    name = "myosuite_oracle_probe",
    srcs = ["myosuite/myosuite_oracle_probe.py"],
    data = select({
        # Linux oracle rendering uses the pinned pip MuJoCo wheel directly.
        "//:linux": [],
        "//conditions:default": ["@mujoco//:mujoco_shared_lib"],
    }) + [
        "//third_party/myosuite:myosuite_generated_json",
        "@myosuite_furniture_sim//:source",
        "@myosuite_mpl_sim//:source",
        "@myosuite_myo_sim//:source",
        "@myosuite_object_sim//:source",
        "@myosuite_source//:source",
        "@myosuite_ycb_sim//:source",
    ],
    imports = ["../.."],
    deps = [
        "//envpool/python:glfw_context",
        oracle_requirement("click"),
        oracle_requirement("dm-control"),
        oracle_requirement("flatten-dict"),
        oracle_requirement("gitpython"),
        oracle_requirement("gymnasium"),
        oracle_requirement("h5py"),
        oracle_requirement("mujoco"),
        oracle_requirement("numpy"),
        oracle_requirement("packaging"),
        oracle_requirement("pillow"),
        oracle_requirement("pink-noise-rl"),
        oracle_requirement("sk-video"),
        oracle_requirement("termcolor"),
    ],
)

py_test(
    name = "metaworld_align_test",
    size = "enormous",
    srcs = ["metaworld/metaworld_align_test.py"],
    imports = ["../.."],
    deps = [
        ":metaworld",
        ":metaworld_registration",
        "@metaworld_assets//:metaworld_oracle",
        requirement("absl-py"),
        requirement("cloudpickle"),
        requirement("gymnasium"),
        requirement("imageio"),
        requirement("mujoco"),
        requirement("numpy"),
        requirement("scipy"),
        requirement("typing-extensions"),
    ],
)

py_test(
    name = "metaworld_render_test",
    size = "enormous",
    srcs = ["metaworld/metaworld_render_test.py"],
    imports = ["../.."],
    deps = [
        ":metaworld",
        ":metaworld_registration",
        "//envpool/python:glfw_context",
        "@metaworld_assets//:metaworld_oracle",
        requirement("absl-py"),
        requirement("gymnasium"),
        requirement("imageio"),
        requirement("mujoco"),
        requirement("numpy"),
        requirement("scipy"),
        requirement("typing-extensions"),
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
        requirement("imageio"),
        requirement("mujoco"),
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
        ":metaworld",
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
