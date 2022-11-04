load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "box2d_env",
    srcs = [
        "bipedal_walker_env.cc",
        "lunar_lander_env.cc",
        "utils.cc",
    ],
    hdrs = [
        "bipedal_walker.h",
        "bipedal_walker_env.h",
        "lunar_lander_continuous.h",
        "lunar_lander_discrete.h",
        "lunar_lander_env.h",
        "utils.h",
    ],
    deps = [
        "//envpool/core:async_envpool",
        "@box2d",
    ],
)

pybind_extension(
    name = "box2d_envpool",
    srcs = ["box2d_envpool.cc"],
    deps = [
        ":box2d_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "box2d",
    srcs = ["__init__.py"],
    data = [":box2d_envpool.so"],
    deps = ["//envpool/python:api"],
)

py_library(
    name = "box2d_registration",
    srcs = ["registration.py"],
    deps = [
        "//envpool:registration",
    ],
)

py_test(
    name = "box2d_deterministic_test",
    size = "enormous",
    srcs = ["box2d_deterministic_test.py"],
    deps = [
        ":box2d",
        ":box2d_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "box2d_correctness_test",
    size = "enormous",
    srcs = ["box2d_correctness_test.py"],
    deps = [
        ":box2d",
        ":box2d_registration",
        requirement("absl-py"),
        requirement("gym"),
        requirement("box2d"),
        requirement("pygame"),
        requirement("numpy"),
    ],
)
