load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

genrule(
    name = "gen_mujoco_xml",
    srcs = ["@mujoco_xml"],
    outs = ["assets"],
    cmd = "mkdir -p $(OUTS) && cp $(SRCS) $(OUTS)",
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

pybind_extension(
    name = "mujoco_envpool",
    srcs = ["mujoco_envpool.cc"],
    deps = [
        ":ant",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "mujoco",
    srcs = ["__init__.py"],
    data = [
        ":gen_mujoco_xml",
        ":gen_mujoco_so",
        ":mujoco_envpool.so",
    ],
    deps = ["//envpool/python:api"],
)

py_test(
    name = "mujoco_test",
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
