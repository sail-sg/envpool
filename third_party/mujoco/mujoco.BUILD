package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mujoco_lib",
    srcs = glob(["lib/*"]),
    hdrs = glob(["include/mujoco/*.h",
                 "sample/*.h"]),
    includes = [
        "include",
        "include/mujoco",
    ],
    linkopts = ["-Wl,-rpath,'$$ORIGIN'"],
    linkstatic = 0,
)

filegroup(
    name = "mujoco_so",
    srcs = ["lib/libmujoco.so.2.2.0"],
)
