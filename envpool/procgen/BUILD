load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "procgen",
    srcs = ["__init__.py"],
    data = [":procgen_envpool.so"],
    deps = ["//envpool/python:api"],
)

cc_library(
    name = "third_party_procgen",
    hdrs = ["third_party_procgen.h"],
    deps = [
        "@procgen",
    ],
)

cc_library(
    name = "procgen_h",
    hdrs = ["procgen.h"],
    deps = [
        ":third_party_procgen",
        "//envpool/core:async_envpool",
        "//envpool/core:env",
        "//envpool/core:env_spec",
        "@procgen",
    ],
)

cc_test(
    name = "procgen_test",
    srcs = ["procgen_test.cc"],
    deps = [
        ":procgen_h",
        "@com_google_googletest//:gtest_main",
    ],
)

py_test(
    name = "procgen_py_test",
    srcs = ["procgen_py_test.py"],
    data = [":procgen_envpool.so"],
    deps = [
        ":procgen",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("opencv-python-headless"),
    ],
)

py_library(
    name = "procgen_registration",
    srcs = ["registration.py"],
    deps = [
        "//envpool:registration",
    ],
)

pybind_extension(
    name = "procgen_envpool",
    srcs = [
        "procgen.cc",
    ],
    linkopts = [
        "-ldl",
    ],
    deps = [
        ":procgen_h",
        "//envpool/core:py_envpool",
        "@procgen",
    ],
)
