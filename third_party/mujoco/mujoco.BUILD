package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mujoco_lib",
    srcs = glob(["lib/*"]),
    hdrs = glob([
        "include/mujoco/*.h",
        "sample/*.h",
    ]),
    includes = [
        "include",
        "include/mujoco",
        "sample",
    ],
    linkopts = [
        "-Wl,-rpath,'$$ORIGIN'",
        "-lOSMesa",
    ],
    linkstatic = 0,
)

filegroup(
    name = "mujoco_so",
    srcs = ["lib/libmujoco.so.2.2.1"],
)
