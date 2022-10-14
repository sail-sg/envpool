load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "classic_control_env",
    hdrs = [
        "acrobot.h",
        "cartpole.h",
        "mountain_car.h",
        "mountain_car_continuous.h",
        "pendulum.h",
    ],
    deps = [
        "//envpool/core:async_envpool",
    ],
)

pybind_extension(
    name = "classic_control_envpool",
    srcs = ["classic_control.cc"],
    deps = [
        ":classic_control_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "classic_control",
    srcs = ["__init__.py"],
    data = [":classic_control_envpool.so"],
    deps = ["//envpool/python:api"],
)

py_test(
    name = "classic_control_test",
    size = "enormous",
    srcs = ["classic_control_test.py"],
    deps = [
        ":classic_control",
        requirement("absl-py"),
        requirement("dm_env"),
        requirement("gym>=0.26.0"),
        requirement("numpy"),
        requirement("pygame"),
    ],
)

py_library(
    name = "classic_control_registration",
    srcs = ["registration.py"],
    deps = [
        "//envpool:registration",
    ],
)
