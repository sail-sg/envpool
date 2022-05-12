load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "car_dynamics",
    srcs = ["car_dynamics.cc"],
    hdrs = ["car_dynamics.h"],
    deps = [
        "@box2d",
    ],
)

cc_library(
    name = "car_racing",
    hdrs = ["car_racing.h"],
    deps = [
        ":car_dynamics",
        "//envpool/core:async_envpool",
        "@box2d",
    ],
)

pybind_extension(
    name = "box2d_envpool",
    srcs = [
        "box2d_envpool.cc",
    ],
    deps = [
        ":car_racing",
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
