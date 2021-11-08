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
        "@concurrentqueue",
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
    name = "circular_buffer",
    hdrs = ["circular_buffer.h"],
    deps = [
        "@concurrentqueue",
    ],
)

cc_test(
    name = "circular_buffer_test",
    srcs = ["circular_buffer_test.cc"],
    deps = [
        ":circular_buffer",
        "@com_github_google_glog//:glog",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "state_buffer_queue",
    hdrs = ["state_buffer_queue.h"],
    deps = [
        ":circular_buffer",
        ":spec",
        ":state_buffer",
    ],
)

cc_test(
    name = "state_buffer_queue_test",
    srcs = ["state_buffer_queue_test.cc"],
    deps = [
        ":state_buffer_queue",
        "@com_google_googletest//:gtest_main",
        "@threadpool",
    ],
)

cc_library(
    name = "action_buffer_queue",
    hdrs = ["action_buffer_queue.h"],
    deps = [
        ":array",
        "@concurrentqueue",
    ],
)

cc_test(
    name = "action_buffer_queue_test",
    srcs = ["action_buffer_queue_test.cc"],
    deps = [
        ":action_buffer_queue",
        ":dict",
        ":spec",
        "@com_google_googletest//:gtest_main",
        "@threadpool",
    ],
)

cc_library(
    name = "env_spec",
    hdrs = ["env_spec.h"],
    deps = [
        ":array",
        ":dict",
    ],
)

cc_library(
    name = "env",
    hdrs = ["env.h"],
    deps = [
        ":spec",
        ":state_buffer_queue",
    ],
)

cc_library(
    name = "envpool",
    hdrs = ["envpool.h"],
    deps = [
        ":env_spec",
    ],
)

cc_library(
    name = "async_envpool",
    hdrs = ["async_envpool.h"],
    deps = [
        ":action_buffer_queue",
        ":array",
        ":env",
        ":envpool",
        ":spec",
        ":state_buffer_queue",
        "@threadpool",
    ],
)

cc_library(
    name = "dummy_async_envpool",
    hdrs = ["dummy_async_envpool.h"],
    deps = [
        ":async_envpool",
    ],
)

cc_test(
    name = "dummy_async_envpool_test",
    srcs = ["dummy_async_envpool_test.cc"],
    deps = [
        ":dummy_async_envpool",
        "@com_google_googletest//:gtest_main",
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

pybind_extension(
    name = "dummy_async_py_envpool",
    srcs = [
        "dummy_async_py_envpool.cc",
    ],
    linkopts = [
        "-ldl",
    ],
    deps = [
        ":dummy_async_envpool",
        ":py_envpool",
    ],
)

py_test(
    name = "dummy_async_py_envpool_test",
    srcs = ["dummy_async_py_envpool_test.py"],
    data = ["dummy_async_py_envpool.so"],
    deps = [
        requirement("numpy"),
        requirement("absl-py"),
    ],
)
