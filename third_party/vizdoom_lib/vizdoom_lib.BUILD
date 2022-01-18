package(default_visibility = ["//visibility:public"])

load("@envpool//third_party:common.bzl", "template_rule")

template_rule(
    name = "vizdoom_version",
    src = "src/lib/ViZDoomVersion.h.in",
    out = "ViZDoomVersion.h",
    substitutions = {
        "@ViZDoom_VERSION_ID@": "1111",
        "@ViZDoom_VERSION_STR@": "1.1.11",
    },
)

cc_library(
    name = "vizdoom_lib",
    srcs = glob([
        "include/*.h",
        "src/lib/*.h",
        "src/lib/*.cpp",
        "src/lib/boost/**/*.hpp",
    ]) + [":vizdoom_version"],
    hdrs = ["include/ViZDoom.h"],
    linkopts = ["-ldl"],
    includes = [
        "include",
        "src/lib",
    ],
    deps = [
        "@boost//:asio",
        "@boost//:filesystem",
        "@boost//:interprocess",
        "@boost//:iostreams",
        "@boost//:process",
        "@boost//:random",
        "@boost//:thread",
    ],
)

filegroup(
    name = "vizdoom_maps",
    srcs = glob(["scenarios/*"]),
)
