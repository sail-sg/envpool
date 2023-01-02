package(default_visibility = ["//visibility:public"])

filegroup(
    name = "procgen_assets",
    srcs = [
        "data/assets/kenney",
        "data/assets/kenney-abstract",
        "data/assets/misc_assets",
        "data/assets/platform_backgrounds",
        "data/assets/platform_backgrounds_2",
        "data/assets/platformer",
        "data/assets/space_backgrounds",
        "data/assets/topdown_backgrounds",
        "data/assets/water_backgrounds",
    ],
)

cc_library(
    name = "procgen",
    srcs = glob(["src/**/*.cpp"]) + glob(["src/*.h"]),
    hdrs = glob(["src/*.h"]),
    copts = [
        "-fpic",
    ],
    deps = [
        "@gym3_libenv//:gym3_libenv_header",
        "@qt//:qt_core",
        "@qt//:qt_gui",
    ],
    strip_include_prefix = "src",
)
