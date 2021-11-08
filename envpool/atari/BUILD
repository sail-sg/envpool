load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

genrule(
    name = "gen_atari_roms",
    srcs = ["@atari_roms//:roms"],
    outs = ["atari_roms"],
    cmd = "cp -r $(SRCS) $(OUTS)",
)

py_library(
    name = "atari",
    srcs = ["__init__.py"],
    data = [":atari_envpool.so"],
    deps = ["//envpool/python:api"],
)

cc_library(
    name = "atari_env",
    hdrs = ["atari_env.h"],
    data = [
        ":gen_atari_roms",
    ],
    deps = [
        "//envpool/core:async_envpool",
        "//envpool/utils:resize",
        "@ale//:ale_interface",
    ],
)

cc_test(
    name = "atari_env_test",
    srcs = ["atari_env_test.cc"],
    deps = [
        ":atari_env",
        "@com_google_googletest//:gtest_main",
    ],
)

pybind_extension(
    name = "atari_envpool",
    srcs = [
        "atari_envpool.cc",
    ],
    linkopts = [
        "-ldl",
    ],
    deps = [
        ":atari_env",
        "//envpool/core:py_envpool",
    ],
)

py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    deps = [
        ":atari",
        "//envpool/python",
        requirement("numpy"),
    ],
)

py_test(
    name = "atari_envpool_test",
    srcs = ["atari_envpool_test.py"],
    deps = [
        ":atari",
        requirement("numpy"),
        requirement("dm-env"),
        requirement("absl-py"),
    ],
)
