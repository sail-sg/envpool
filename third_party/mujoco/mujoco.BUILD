package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mujoco_lib",
    srcs = glob(["src/**/*.h"]) + glob(["src/**/*.c"]) + glob(["src/**/*.cc"]) + glob(["src/**/*.inc"]),
    hdrs = glob(["include/mujoco/*.h"]),
    includes = [
        "include",
        "include/mujoco",
        "src",
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
