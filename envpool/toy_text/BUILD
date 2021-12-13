load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "catch",
    hdrs = ["catch.h"],
    deps = [
        "//envpool/core:async_envpool",
    ],
)

pybind_extension(
    name = "toy_text_envpool",
    srcs = [
        "toy_text.cc",
    ],
    deps = [
        ":catch",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "toy_text",
    srcs = ["__init__.py"],
    data = [":toy_text_envpool.so"],
    deps = ["//envpool/python:api"],
)

py_test(
    name = "toy_text_test",
    srcs = ["toy_text_test.py"],
    deps = [
        ":toy_text",
        requirement("absl-py"),
        requirement("dm_env"),
        requirement("gym"),
        requirement("numpy"),
    ],
)

py_library(
    name = "toy_text_registration",
    srcs = ["registration.py"],
    deps = [
        "//envpool:registration",
    ],
)
