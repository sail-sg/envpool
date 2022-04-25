package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mujoco_lib",
    srcs = glob(["lib/*"]),
    hdrs = glob(["include/*.h"]),
    includes = [
        "include/",
    ],
    linkstatic = 0,
)

filegroup(
    name = "mujoco_so",
    srcs = ["lib/libmujoco.so.2.1.5"],
)
