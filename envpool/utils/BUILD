package(default_visibility = ["//visibility:public"])

cc_library(
    name = "resize",
    hdrs = ["resize.h"],
    deps = [
        "//envpool/core:array",
        "@opencv",
    ],
)

cc_test(
    name = "resize_test",
    srcs = ["resize_test.cc"],
    deps = [
        ":resize",
        "@com_google_googletest//:gtest_main",
    ],
)
