# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    name = "shared_thread_pool_h",
    hdrs = ["shared_thread_pool.h"],
    deps = [":action_buffer_queue"],
)

cc_library(
    name = "async_envpool",
    hdrs = ["async_envpool.h"],
    deps = [
        ":action_buffer_queue",
        ":array",
        ":env",
        ":envpool",
        ":shared_thread_pool_h",
        ":spec",
        ":state_buffer_queue",
        "@threadpool",
    ],
)

cc_library(
    name = "xla_template",
    hdrs = ["xla_template.h"],
    linkopts = [
        "-ldl",
        "-lrt",
    ],
    deps = [
        "@cuda//:cudart_static",
    ],
)

cc_library(
    name = "xla",
    hdrs = ["xla.h"],
    deps = [
        ":array",
        ":xla_template",
        "@cuda//:cudart_static",
    ],
)

pybind_library(
    name = "py_envpool",
    hdrs = ["py_envpool.h"],
    deps = [
        ":envpool",
        ":shared_thread_pool_h",
        ":xla",
    ],
)

pybind_extension(
    name = "shared_thread_pool",
    srcs = ["shared_thread_pool.cc"],
    deps = [
        ":shared_thread_pool_h",
    ],
)

py_library(
    name = "core",
    srcs = ["__init__.py"],
    data = [":shared_thread_pool.so"],
)
