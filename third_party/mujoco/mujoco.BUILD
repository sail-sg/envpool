# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mujoco_lib",
    srcs = (
        glob(["src/cc/*.h"]) + glob([
            "src/engine/*.c",
            "src/engine/*.cc",
            "src/engine/*.h",
        ]) + glob([
            "src/thread/*.c",
            "src/thread/*.cc",
            "src/thread/*.h",
        ]) + glob([
            "src/render/classic/*.c",
            "src/render/classic/*.cc",
            "src/render/classic/*.h",
        ]) + glob([
            "src/render/classic/glad/*",
        ]) + glob([
            "src/user/*.c",
            "src/user/*.cc",
            "src/user/*.h",
        ]) + glob([
            "src/xml/*.c",
            "src/xml/*.cc",
            "src/xml/*.h",
        ])
    ),
    hdrs = glob([
        "include/mujoco/*.h",
        "include/mujoco/experimental/**/*.h",
        "src/render/classic/**/*.h",
        "src/render/classic/**/*.inc",
    ]),
    copts = [
        "-DCCD_STATIC_DEFINE",
    ] + select({
        "@envpool//:windows": [],
        "//conditions:default": [
            "-D_GNU_SOURCE",
            "-Wno-int-in-bool-context",
            "-Wno-maybe-uninitialized",
            "-Wno-sign-compare",
            "-Wno-stringop-overflow",
            "-Wno-stringop-truncation",
        ],
    }),
    cxxopts = select({
        "@envpool//:windows": ["/std:c++20"],
        "//conditions:default": ["-std=c++20"],
    }),
    defines = ["MJ_STATIC"],
    includes = [
        "include",
        "include/mujoco",
        "src",
    ],
    linkstatic = 1,
    linkopts = select({
        "@envpool//:linux": [
            "-ldl",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "@ccd",
        "@lodepng",
        "@marchingcubecpp",
        "@qhull",
        "@tinyobjloader",
        "@tinyxml2",
    ],
)
