load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "wrapper",
    hdrs = ["wrapper.h"],
    deps = [
        "//envpool/core:async_envpool",
    ],
)

pybind_extension(
    name = "gym_envpool",
    srcs = ["gym.cc"],
    deps = [
        ":wrapper",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "gym",
    srcs = ["__init__.py"],
    data = [":gym_envpool.so"],
    deps = ["//envpool/python:api"],
)

py_test(
    name = "gym_test",
    srcs = ["gym_test.py"],
    deps = [
        ":gym",
        requirement("absl-py"),
        requirement("gym"),
        requirement("numpy"),
    ],
)
