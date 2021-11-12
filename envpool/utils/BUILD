package(default_visibility = ["//visibility:public"])

cc_library(
    name = "image_process",
    hdrs = ["image_process.h"],
    deps = [
        "//envpool/core:array",
        "@opencv",
    ],
)

cc_test(
    name = "image_process_test",
    srcs = ["image_process_test.cc"],
    deps = [
        ":image_process",
        "@com_google_googletest//:gtest_main",
    ],
)
