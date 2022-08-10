load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "football_env",
    hdrs = ["football_env.h"],
    deps = [
        "//envpool/core:async_envpool",
        "@gfootball_engine//:game_env",
        "//envpool/football:config",
        "//envpool/football:football_action_set",
    ],
)

cc_library(
    name = "config",
    hdrs = ["config.h"],
    deps = [
        "@gfootball_engine//:game_env",
    ],
)

cc_library(
    name = "football_action_set",
    hdrs = ["football_action_set.h"],
    deps = [
        "//envpool/football:config",
        "@gfootball_engine//:game_env",
    ],
)

pybind_extension(
    name = "football_envpool",
    srcs = [
        "football_envpool.cc",
    ],
    deps = [
        "//envpool/football:football_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "football",
    srcs = ["__init__.py"],
    deps = ["//envpool/python:api"]
)

py_test(
    name = "football_envpool_test",
    size = "enormous",
    srcs = ["football_envpool_test.py"],
    deps = [
        ":football",
        requirement("numpy"),
        requirement("jax"),
        requirement("absl-py"),
    ],
)

cc_test(
    name = "football_env_test",
    size = "enormous",
    srcs = ["football_env_test.cc"],
    deps = [
        ":football_env",
        "@com_google_googletest//:gtest_main",
    ],
)

