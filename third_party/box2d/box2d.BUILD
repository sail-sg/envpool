package(default_visibility = ["//visibility:public"])

cc_library(
    name = "box2d",
    srcs = glob([
        "Box2D/**/*.h",
        "Box2D/**/*.cpp",
        "Box2D/**/**/*.h",
        "Box2D/**/**/*.cpp",
    ]),
    hdrs = ["Box2D/Box2D.h"],
    includes = ["."],
    linkopts = [
        "-ldl",
    ],
    linkstatic = 1,
)
