load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

genrule(
    name = "gen_atari_roms",
    srcs = ["@atari_roms//:roms"],
    outs = ["roms"],
    cmd = "mkdir -p $(OUTS) && cp $(SRCS) $(OUTS)",
)

genrule(
    name = "gen_pretrain_weight",
    srcs = [
        "@pretrain_weight//:pth",
    ],
    outs = [
        "policy-pong.pth",
        "policy-breakout.pth",
    ],
    cmd = "cp -r $(SRCS) $(@D)",
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
        "//envpool/utils:image_process",
        "@ale//:ale_interface",
    ],
)

cc_test(
    name = "atari_env_test",
    size = "enormous",
    srcs = ["atari_env_test.cc"],
    deps = [
        ":atari_env",
        "@com_google_googletest//:gtest_main",
    ],
)

py_library(
    name = "atari_network",
    srcs = ["atari_network.py"],
    deps = [
        requirement("numpy"),
        requirement("torch"),
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
    size = "enormous",
    srcs = ["atari_envpool_test.py"],
    deps = [
        ":atari",
        requirement("numpy"),
        requirement("jax"),
        requirement("dm-env"),
        requirement("absl-py"),
        requirement("opencv-python-headless"),
    ],
)

py_test(
    name = "atari_pretrain_test",
    size = "enormous",
    srcs = ["atari_pretrain_test.py"],
    data = [
        ":gen_pretrain_weight",
    ],
    deps = [
        ":atari",
        ":atari_network",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("tianshou"),
        requirement("tqdm"),
        requirement("torch"),
    ],
)

py_library(
    name = "atari_registration",
    srcs = ["registration.py"],
    deps = [
        "//envpool:registration",
    ],
)
