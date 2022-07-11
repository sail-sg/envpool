package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cudart_static",
    srcs = ["lib64/libcudart_static.a"],
    hdrs = glob([
        "include/*.h",
        "include/**/*.h",
    ]),
    strip_include_prefix = "include",
)
