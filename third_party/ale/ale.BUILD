package(default_visibility = ["//visibility:public"])

load("@envpool//third_party:common.bzl", "template_rule")

cc_library(
    name = "irregular_files",
    hdrs = glob([
        "src/**/*.def",
        "src/**/*.ins",
    ]),
)

template_rule(
    name = "ale_version",
    src = "src/version.hpp.in",
    out = "version.hpp",
    substitutions = {
        "@ALE_VERSION@": "0.7.5",
        "@ALE_VERSION_MAJOR@": "0",
        "@ALE_VERSION_MINOR@": "7",
        "@ALE_VERSION_PATCH@": "5",
        "@ALE_VERSION_GIT_SHA@": "978d2ce25665338ba71e45d32fff853b17c15f2e",
    },
)

cc_library(
    name = "ale_interface",
    srcs = glob(
        [
            "src/**/*.h",
            "src/**/*.hpp",
            "src/**/*.hxx",
            "src/**/*.c",
            "src/**/*.cpp",
            "src/**/*.cxx",
        ],
        exclude = [
            "src/python/*",
        ],
    ) + [
        ":ale_version",
    ],
    hdrs = ["src/ale_interface.hpp"],
    linkopts = [
        "-ldl",
    ],
    includes = [
        "src",
        "src/common",
        "src/emucore",
        "src/environment",
        "src/games",
        "src/games/supported",
    ],
    linkstatic = 0,
    deps = [
        ":irregular_files",
        "@zlib",
    ],
)
