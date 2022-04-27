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

cc_library(
    name = "mujoco_env",
    hdrs = ["mujoco_env.h"],
    data = [
        ":gen_mujoco_xml",
    ],
    deps = [
        "@mujoco//:mujoco_lib",
    ],
)

cc_library(
    name = "ant",
    hdrs = ["ant.h"],
    deps = [
        ":mujoco_env",
        "//envpool/core:async_envpool",
    ],
)

cc_library(
    name = "half_cheetah",
    hdrs = ["half_cheetah.h"],
    deps = [
        ":mujoco_env",
        "//envpool/core:async_envpool",
    ],
)

cc_library(
    name = "hopper",
    hdrs = ["hopper.h"],
    deps = [
        ":mujoco_env",
        "//envpool/core:async_envpool",
    ],
)

cc_library(
    name = "humanoid",
    hdrs = ["humanoid.h"],
    deps = [
        ":mujoco_env",
        "//envpool/core:async_envpool",
    ],
)

cc_library(
    name = "humanoid_standup",
    hdrs = ["humanoid_standup.h"],
    deps = [
        ":mujoco_env",
        "//envpool/core:async_envpool",
    ],
)

cc_library(
    name = "inverted_double_pendulum",
    hdrs = ["inverted_double_pendulum.h"],
    deps = [
        ":mujoco_env",
        "//envpool/core:async_envpool",
    ],
)

cc_library(
    name = "inverted_pendulum",
    hdrs = ["inverted_pendulum.h"],
    deps = [
        ":mujoco_env",
        "//envpool/core:async_envpool",
    ],
)

cc_library(
    name = "pusher",
    hdrs = ["pusher.h"],
    deps = [
        ":mujoco_env",
        "//envpool/core:async_envpool",
    ],
)

cc_library(
    name = "reacher",
    hdrs = ["reacher.h"],
    deps = [
        ":mujoco_env",
        "//envpool/core:async_envpool",
    ],
)

cc_library(
    name = "swimmer",
    hdrs = ["swimmer.h"],
    deps = [
        ":mujoco_env",
        "//envpool/core:async_envpool",
    ],
)

cc_library(
    name = "walker2d",
    hdrs = ["walker2d.h"],
    deps = [
        ":mujoco_env",
        "//envpool/core:async_envpool",
    ],
)

pybind_extension(
    name = "mujoco_envpool",
    srcs = ["mujoco_envpool.cc"],
    deps = [
        ":ant",
        ":half_cheetah",
        ":hopper",
        ":humanoid",
        ":humanoid_standup",
        ":inverted_double_pendulum",
        ":inverted_pendulum",
        ":pusher",
        ":reacher",
        ":swimmer",
        ":walker2d",
        "//envpool/core:py_envpool",
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
