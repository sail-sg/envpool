load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "darwin",
    constraint_values = ["@platforms//os:macos"],
)

cc_library(
    name = "sdl2_ttf",
    srcs = ["SDL_ttf.c"],
    hdrs = ["SDL_ttf.h"],
    include_prefix = "SDL2",
    linkopts = select({
        ":darwin": [
            "-L/opt/homebrew/opt/libpng/lib",
            "-L/usr/local/opt/libpng/lib",
            "-lpng16",
        ],
        "//conditions:default": ["-lpng"],
    }),
    strip_include_prefix = ".",
    deps = [
        "@freetype_system//:freetype",
        "@sdl2",
    ],
)
