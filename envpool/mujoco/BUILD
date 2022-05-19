load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

genrule(
    name = "gen_mujoco_gym_xml",
    srcs = ["@mujoco_gym_xml"],
    outs = ["assets_gym"],
    cmd = "mkdir -p $(OUTS) && cp -r $(SRCS) $(OUTS)",
)

genrule(
    name = "gen_mujoco_dmc_xml",
    srcs = ["@mujoco_dmc_xml"],
    outs = ["assets_dmc"],
    cmd = "mkdir -p $(OUTS) && cp -r $(SRCS) $(OUTS)",
)

genrule(
    name = "gen_mujoco_so",
    srcs = ["@mujoco//:mujoco_so"],
    outs = ["libmujoco.so.2.1.5"],
    cmd = "cp $< $@",
)

cc_library(
    name = "mujoco_gym_env",
    hdrs = [
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
    ],
    data = [
        ":gen_mujoco_gym_xml",
    ],
    deps = [
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
    hdrs = [
        "dmc/acrobot.h",
        "dmc/ball_in_cup.h",
        "dmc/cheetah.h",
        "dmc/finger.h",
        "dmc/hopper.h",
        "dmc/humanoid.h",
        "dmc/mujoco_env.h",
        "dmc/pendulum.h",
        "dmc/point_mass.h",
        "dmc/reacher.h",
        "dmc/utils.h",
        "dmc/walker.h",
    ],
    data = [":gen_mujoco_dmc_xml"],
    deps = [
        "//envpool/core:async_envpool",
        "@mujoco//:mujoco_lib",
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
        ":gen_mujoco_so",
        ":mujoco_dmc_envpool.so",
    ],
    deps = ["//envpool/python:api"],
)

py_library(
    name = "mujoco_gym",
    srcs = ["gym/__init__.py"],
    data = [
        ":gen_mujoco_gym_xml",
        ":gen_mujoco_so",
        ":mujoco_gym_envpool.so",
    ],
    deps = ["//envpool/python:api"],
)

py_test(
    name = "mujoco_gym_deterministic_test",
    size = "enormous",
    srcs = ["gym/mujoco_gym_deterministic_test.py"],
    deps = [
        ":mujoco_gym",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("gym"),
    ],
)

py_test(
    name = "mujoco_gym_align_test",
    size = "enormous",
    srcs = ["gym/mujoco_gym_align_test.py"],
    deps = [
        ":mujoco_gym",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("gym"),
        requirement("mujoco"),
        requirement("mjc_mwe"),
    ],
)

py_test(
    name = "mujoco_dmc_deterministic_test",
    size = "enormous",
    srcs = ["dmc/mujoco_dmc_deterministic_test.py"],
    deps = [
        ":mujoco_dmc",
        requirement("numpy"),
        requirement("absl-py"),
    ],
)

py_test(
    name = "mujoco_dmc_align_test",
    size = "enormous",
    srcs = ["dmc/mujoco_dmc_align_test.py"],
    deps = [
        ":mujoco_dmc",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("mujoco"),
        requirement("dm-env"),
        requirement("dm-control"),
    ],
)

py_library(
    name = "mujoco_dmc_registration",
    srcs = ["dmc/registration.py"],
    deps = ["//envpool:registration"],
)

py_library(
    name = "mujoco_gym_registration",
    srcs = ["gym/registration.py"],
    deps = ["//envpool:registration"],
)
