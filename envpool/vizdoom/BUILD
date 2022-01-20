load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

genrule(
    name = "gen_pretrain_weight",
    srcs = [
        "@pretrain_weight//:pth",
    ],
    outs = [
        "policy-d1.pth",
        "policy-d3.pth",
    ],
    cmd = "cp $(SRCS) $(@D)",
)

genrule(
    name = "gen_vizdoom_maps",
    srcs = [
        "@vizdoom_lib//:vizdoom_maps",
        "@vizdoom_extra_maps//:vizdoom_maps",
    ],
    outs = ["maps"],
    cmd = "mkdir -p $(OUTS) && cp $(SRCS) $(OUTS)",
)

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
    deps = ["@vizdoom_lib"],
)

cc_library(
    name = "vizdoom_env",
    hdrs = ["vizdoom_env.h"],
    deps = [
        ":utils",
        "//envpool/core:async_envpool",
        "//envpool/utils:image_process",
    ],
)

pybind_extension(
    name = "vizdoom_envpool",
    srcs = ["vizdoom_envpool.cc"],
    linkopts = [
        "-ldl",
    ],
    deps = [
        ":vizdoom_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "vizdoom",
    srcs = ["__init__.py"],
    data = [
        "//envpool/vizdoom/bin:vizdoom_bin",
        "//envpool/vizdoom/bin:vizdoom_pk3",
        "//envpool/vizdoom/bin:freedoom",
        ":gen_vizdoom_maps",
        ":vizdoom_envpool.so",
    ],
    deps = ["//envpool/python:api"],
)

py_test(
    name = "vizdoom_test",
    srcs = ["vizdoom_test.py"],
    deps = [
        ":vizdoom",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("opencv-python-headless"),
    ],
)

py_test(
    name = "vizdoom_pretrain_test",
    srcs = ["vizdoom_pretrain_test.py"],
    data = [":gen_pretrain_weight"],
    deps = [
        ":vizdoom",
        "//envpool/atari:atari_network",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("tianshou"),
        requirement("torch"),
    ],
)

py_library(
    name = "vizdoom_registration",
    srcs = ["registration.py"],
    deps = ["//envpool:registration"],
)
