load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "sdl2_gfx",
    srcs = ["SDL2_rotozoom.c"],
    hdrs = ["SDL2_rotozoom.h"],
    include_prefix = "SDL2",
    strip_include_prefix = ".",
    deps = ["@sdl2"],
)
