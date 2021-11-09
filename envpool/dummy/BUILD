load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

py_library(
    name = "dummy",
    srcs = ["__init__.py"],
    data = [":dummy_envpool.so"],
    deps = ["//envpool/python:api"],
)

cc_library(
    name = "dummy_envpool",
    hdrs = ["dummy_envpool.h"],
    deps = [
        "//envpool/core:async_envpool",
        "//envpool/core:env",
        "//envpool/core:env_spec",
    ],
)

cc_test(
    name = "dummy_envpool_test",
    srcs = ["dummy_envpool_test.cc"],
    deps = [
        ":dummy_envpool",
        "@com_google_googletest//:gtest_main",
    ],
)

py_test(
    name = "dummy_py_envpool_test",
    srcs = ["dummy_py_envpool_test.py"],
    data = ["dummy_envpool.so"],
    deps = [
        requirement("numpy"),
        requirement("absl-py"),
    ],
)

pybind_extension(
    name = "dummy_envpool",
    srcs = [
        "dummy_envpool.cc",
    ],
    linkopts = [
        "-ldl",
    ],
    deps = [
        ":dummy_envpool",
        "//envpool/core:py_envpool",
    ],
)
