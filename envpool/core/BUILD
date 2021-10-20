load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "type_utils",
    hdrs = ["type_utils.h"],
)

cc_library(
    name = "tuple_utils",
    hdrs = ["tuple_utils.h"],
)

cc_library(
    name = "spec",
    hdrs = ["spec.h"],
    deps = [
        "@com_github_google_glog//:glog",
    ],
)

cc_library(
    name = "array",
    hdrs = ["array.h"],
    deps = [
        ":spec",
        "@com_github_google_glog//:glog",
    ],
)

cc_library(
    name = "dict",
    hdrs = ["dict.h"],
    deps = [
        ":array",
        ":spec",
        ":tuple_utils",
        ":type_utils",
        "@com_github_google_glog//:glog",
    ],
)

cc_test(
    name = "dict_test",
    srcs = ["dict_test.cc"],
    deps = [
        ":dict",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "state_buffer",
    hdrs = ["state_buffer.h"],
    deps = [
        ":array",
        ":dict",
        ":spec",
    ],
)

cc_test(
    name = "state_buffer_test",
    srcs = [
        "state_buffer_test.cc",
    ],
    deps = [
        ":state_buffer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "buffer_queue",
    hdrs = ["buffer_queue.h"],
    deps = [
        ":spec",
        ":state_buffer",
    ],
)

cc_test(
    name = "buffer_queue_test",
    srcs = ["buffer_queue_test.cc"],
    deps = [
        ":buffer_queue",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "envpool",
    hdrs = ["envpool.h"],
    deps = [
        ":array",
        ":dict",
        ":spec",
    ],
)

pybind_library(
    name = "py_envpool",
    hdrs = ["py_envpool.h"],
    deps = [
        ":envpool",
    ],
)

pybind_extension(
    name = "dummy_envpool",
    srcs = [
        "dummy_envpool.cc",
    ],
    linkopts = [
        "-ldl",
    ],
    deps = [
        ":dict",
        ":py_envpool",
        ":spec",
    ],
)

py_test(
    name = "dummy_envpool_test",
    srcs = ["dummy_envpool_test.py"],
    data = ["dummy_envpool.so"],
    deps = [
        requirement("numpy"),
        requirement("absl-py"),
    ],
)
