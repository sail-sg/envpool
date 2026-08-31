# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE.md"])

filegroup(
    name = "licenses",
    srcs = ["LICENSE.md"] + glob(["licenses/*.txt"]),
)

cc_library(
    name = "native",
    srcs = ["warp/native/" + name + ".cpp" for name in [
        "alloc_tracker",
        "apic",
        "bvh",
        "coloring",
        "crt",
        "cuda_util",
        "error",
        "hashgrid",
        "mathdx",
        "mesh",
        "reduce",
        "runlength_encode",
        "scan",
        "sort",
        "sparse",
        "texture",
        "volume",
        "warp",
    ]],
    hdrs = glob([
        "warp/native/**/*.h",
        "warp/native/**/*.hpp",
        "warp/native/**/*.cuh",
    ]),
    copts = select({
        "@envpool//:windows": [
            "/fp:strict",
            "/bigobj",
        ],
        "//conditions:default": ["-ffp-contract=off"],
    }),
    defines = [
        "WP_ENABLE_CUDA=0",
        "WP_ENABLE_CUDA_COMPATIBILITY=0",
        "WP_ENABLE_DEBUG=0",
        "WP_ENABLE_MATHDX=0",
        "WP_ENABLE_TILES_IN_STACK_MEMORY",
        "WP_NATIVE_THREAD_LOCAL",
    ],
    includes = ["warp/native"],
    deps = ["@envpool//third_party/mjlab:warp_generated_headers"],
)
