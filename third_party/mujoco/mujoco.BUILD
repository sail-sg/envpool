package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mujoco_lib",
    srcs = glob(["src/**/*.h"]) + glob(["src/**/*.c"]) + glob(["src/**/*.cc"]) + glob(["src/**/*.inc"]),
    hdrs = glob(["include/mujoco/*.h"]),
    copts = [
        "-mavx",
        "-Wno-int-in-bool-context",
        "-Wno-maybe-uninitialized",
        "-Wno-sign-compare",
        "-Wno-stringop-overflow",
        "-Wno-stringop-truncation",
    ],
    includes = [
        "include",
        "include/mujoco",
        "src",
    ],
    linkopts = [
        "-fuse-ld=gold",
        "-Wl,--gc-sections",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "@ccd",
        "@lodepng",
        "@qhull",
        "@tinyobjloader",
        "@tinyxml2",
    ],
)
