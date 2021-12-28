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

cc_library(
    name = "frozen_lake",
    hdrs = ["frozen_lake.h"],
    deps = [
        "//envpool/core:async_envpool",
    ],
)

cc_library(
    name = "taxi",
    hdrs = ["taxi.h"],
    deps = [
        "//envpool/core:async_envpool",
    ],
)

cc_library(
    name = "nchain",
    hdrs = ["nchain.h"],
    deps = [
        "//envpool/core:async_envpool",
    ],
)

cc_library(
    name = "cliffwalking",
    hdrs = ["cliffwalking.h"],
    deps = [
        "//envpool/core:async_envpool",
    ],
)

cc_library(
    name = "blackjack",
    hdrs = ["blackjack.h"],
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
        ":blackjack",
        ":catch",
        ":cliffwalking",
        ":frozen_lake",
        ":nchain",
        ":taxi",
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
        requirement("scipy"),
    ],
)

py_library(
    name = "toy_text_registration",
    srcs = ["registration.py"],
    deps = [
        "//envpool:registration",
    ],
)
