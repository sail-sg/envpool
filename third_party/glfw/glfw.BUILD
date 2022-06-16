package(default_visibility = ["//visibility:public"])

LINUX_LINKOPTS = []

cc_library(
    name = "glfw",
    hdrs = [
        "include/GLFW/glfw3.h",
        "include/GLFW/glfw3native.h",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:linux_x86_64": LINUX_LINKOPTS,
    }),
    deps = [],
    strip_include_prefix = "include",
)