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

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

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
            "src/ui/*.c",
            "src/ui/*.h",
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
    }) + select({
        # Match upstream MuJoCo's default CMake build on Linux x86_64. The
        # pinned official oracle wheel enables AVX platform SIMD there.
        "@envpool//:linux_x86_64": [
            # CI runs Bazel tests in fastbuild, while the official Python
            # wheel is release-built. Keep MuJoCo's integrator codegen aligned
            # with the wheel instead of compensating with looser oracle checks.
            "-O3",
            "-mavx",
            "-mpclmul",
        ],
        "//conditions:default": [],
    }),
    cxxopts = select({
        "@envpool//:windows": ["/std:c++20"],
        "//conditions:default": ["-std=c++20"],
    }),
    defines = ["MJ_STATIC"] + select({
        "@envpool//:linux_x86_64": ["mjUSEPLATFORMSIMD"],
        "//conditions:default": [],
    }),
    # Coverage instrumentation perturbs MuJoCo's floating-point integrator on
    # Linux enough to invalidate long oracle rollouts. Keep third-party
    # physics code out of EnvPool coverage instead of widening oracle drift.
    features = ["-coverage"],
    includes = [
        "include",
        "include/mujoco",
        "src",
    ],
    linkopts = select({
        "@envpool//:linux": [
            "-ldl",
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    deps = [
        "@ccd",
        "@lodepng",
        "@marchingcubecpp",
        "@qhull",
        "@tinyobjloader",
        "@tinyxml2",
    ],
)

cc_binary(
    name = "libmujoco.so.3.6.0",
    linkopts = select({
        "@envpool//:linux": ["-Wl,-soname,libmujoco.so.3.6.0"],
        "//conditions:default": [],
    }),
    linkshared = True,
    linkstatic = True,
    deps = [":mujoco_shared_export_lib"],
)

filegroup(
    name = "mujoco_shared_lib",
    srcs = [":libmujoco.so.3.6.0"],
)

cc_library(
    name = "mujoco_shared_export_lib",
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
            "src/ui/*.c",
            "src/ui/*.h",
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
    }) + select({
        "@envpool//:linux_x86_64": [
            "-O3",
            "-mavx",
            "-mpclmul",
        ],
        "//conditions:default": [],
    }),
    cxxopts = select({
        "@envpool//:windows": ["/std:c++20"],
        "//conditions:default": ["-std=c++20"],
    }),
    defines = ["MUJOCO_DLL_EXPORTS"] + select({
        "@envpool//:linux_x86_64": ["mjUSEPLATFORMSIMD"],
        "//conditions:default": [],
    }),
    features = ["-coverage"],
    includes = [
        "include",
        "include/mujoco",
        "src",
    ],
    linkopts = select({
        "@envpool//:linux": [
            "-ldl",
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    deps = [
        "@ccd",
        "@lodepng",
        "@marchingcubecpp",
        "@qhull",
        "@tinyobjloader",
        "@tinyxml2",
    ],
    alwayslink = True,
)

cc_library(
    name = "mujoco_obj_decoder_plugin_lib",
    srcs = glob([
        "plugin/obj_decoder/*.cc",
        "plugin/obj_decoder/*.h",
    ]),
    cxxopts = select({
        "@envpool//:windows": ["/std:c++20"],
        "//conditions:default": ["-std=c++20"],
    }),
    defines = ["MJ_STATIC"] + select({
        "@envpool//:windows": ["mjDLLMAIN=MjObjDecoderDllMain"],
        "//conditions:default": [],
    }),
    deps = [
        ":mujoco_lib",
        "@tinyobjloader",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mujoco_stl_decoder_plugin_lib",
    srcs = glob([
        "plugin/stl_decoder/*.cc",
        "plugin/stl_decoder/*.h",
    ]),
    cxxopts = select({
        "@envpool//:windows": ["/std:c++20"],
        "//conditions:default": ["-std=c++20"],
    }),
    defines = ["MJ_STATIC"] + select({
        "@envpool//:windows": ["mjDLLMAIN=MjStlDecoderDllMain"],
        "//conditions:default": [],
    }),
    deps = [
        ":mujoco_lib",
    ],
    alwayslink = 1,
)
