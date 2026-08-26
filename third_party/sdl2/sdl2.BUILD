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
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

exports_files(["include/SDL2/SDL.h"])

config_setting(
    name = "darwin",
    constraint_values = ["@platforms//os:macos"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "sdl2_static",
    generate_args = [
        "-GNinja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        "-DSDL2COMPAT_INSTALL=ON",
        "-DSDL2COMPAT_STATIC=ON",
        "-DSDL2COMPAT_TESTS=OFF",
    ] + select({
        "@envpool//:windows": ["-DSDL2COMPAT_LIBC=ON"],
        "//conditions:default": [],
    }),
    lib_source = ":srcs",
    out_include_dir = "include",
    out_static_libs = select({
        "@envpool//:windows": ["SDL2-static.lib"],
        "//conditions:default": ["libSDL2.a"],
    }),
    visibility = ["//visibility:public"],
    deps = ["@sdl3//:sdl3_static"],
)

cc_library(
    name = "sdl2",
    linkopts = select({
        ":darwin": [
            "-framework CoreVideo",
            "-framework Cocoa",
            "-framework IOKit",
            "-framework ForceFeedback",
            "-framework Carbon",
            "-framework CoreAudio",
            "-framework AudioToolbox",
            "-framework AVFoundation",
            "-framework CoreBluetooth",
            "-framework CoreGraphics",
            "-framework Foundation",
            "-framework CoreServices",
            "-weak_framework GameController",
            "-weak_framework Metal",
            "-weak_framework QuartzCore",
            "-weak_framework CoreHaptics",
        ],
        "@envpool//:windows": [
            "gdi32.lib",
            "imm32.lib",
            "ole32.lib",
            "oleaut32.lib",
            "setupapi.lib",
            "shell32.lib",
            "user32.lib",
            "version.lib",
            "winmm.lib",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":sdl2_static",
        "@sdl3//:sdl3_static",
    ],
)
