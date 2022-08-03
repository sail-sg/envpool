load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "football_env",
    hdrs = ["football_env.h"],
    deps = [
        "//envpool/core:async_envpool",
        "@gfootball_engine//:game_env",
        "config.h",
        "football_action_set.h",
    ],
)


pybind_extension(
    name = "football_envpool",
    srcs = [
        "football_envpool.cc",
    ],
    deps = [
        "football_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "football",
    srcs = ["__init__.py"],
    deps = ["//envpool/python:api"]
)

py_library(
    name = "football_observation_processor",
    srcs = ["football_observation_processor.py"],
    deps = [
        requirement("numpy"),
        requirement("collections"),
        requirement("datetime"),
        requirement("os"),
        requirement("shutil"),
        requirement("tempfile"),
        requirement("timeit"),
        requirement("traceback"),
        requirement("absl"),
        requirement("gfootball"),
        requirement("six"),
    ],
)

