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

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

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
        "//envpool/core:logging",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "jax_random",
    hdrs = ["jax_random.h"],
)

cc_library(
    name = "numpy_random",
    hdrs = ["numpy_random.h"],
    deps = ["//third_party/numpy:random_headers"],
)

cc_library(
    name = "colored_noise",
    hdrs = ["colored_noise.h"],
    deps = [
        ":numpy_random",
        "//third_party/numpy:fft",
    ],
)
