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

'''
genrule(
    name = "gen_vizdoom_lib",
    srcs = ["@vizdoom_lib//:vizdoom_lib"],
    outs = ["lib"],
    cmd = "cp -r $(SRCS) $(OUTS)",
)
'''

py_library(
    name = "atari_network",
    srcs = ["atari_network.py"],
    deps = [
        requirement("numpy"),
        requirement("torch"),
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
        ":atari_network",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("tianshou"),
        requirement("tqdm"),
        requirement("torch"),
    ],
)
