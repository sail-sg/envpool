load("@pip_requirements//:requirements.bzl", "requirement")

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
    deps = [
        "@vizdoom_lib",
    ],
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

py_test(
    name = "vizdoom_test",
    srcs = ["vizdoom_test.py"],
    data = [
        ":gen_pretrain_weight",
        ":gen_vizdoom_maps",
    ],
    deps = [
        "//envpool/atari:atari_network",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("tianshou"),
        requirement("tqdm"),
        requirement("torch"),
    ],
)
