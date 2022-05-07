load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

genrule(
    name = "gen_mujoco_xml",
    srcs = ["@mujoco_xml"],
    outs = ["assets"],
    cmd = "cp -r $(SRCS) $(@D)",
)

genrule(
    name = "gen_mujoco_so",
    srcs = ["@mujoco//:mujoco_so"],
    outs = ["libmujoco.so.2.1.5"],
    cmd = "cp $< $@",
)

pybind_extension(
    name = "mujoco_envpool",
    srcs = [
        "ant.h",
        "half_cheetah.h",
        "hopper.h",
        "humanoid.h",
        "humanoid_standup.h",
        "inverted_double_pendulum.h",
        "inverted_pendulum.h",
        "mujoco_env.h",
        "mujoco_envpool.cc",
        "pusher.h",
        "reacher.h",
        "swimmer.h",
        "walker2d.h",
    ],
    data = [
        ":gen_mujoco_xml",
    ],
    deps = [
        "//envpool/core:async_envpool",
        "//envpool/core:py_envpool",
        "@mujoco//:mujoco_lib",
    ],
)

py_library(
    name = "mujoco",
    srcs = ["__init__.py"],
    data = [
        ":gen_mujoco_so",
        ":gen_mujoco_xml",
        ":mujoco_envpool.so",
    ],
    deps = ["//envpool/python:api"],
)

py_test(
    name = "mujoco_test",
    size = "enormous",
    srcs = ["mujoco_test.py"],
    deps = [
        ":mujoco",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("gym"),
        requirement("mujoco"),
        requirement("mjc_mwe"),
    ],
)

py_library(
    name = "mujoco_registration",
    srcs = ["registration.py"],
    deps = ["//envpool:registration"],
)
