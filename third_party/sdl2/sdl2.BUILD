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

config_setting(
    name = "darwin",
    constraint_values = ["@platforms//os:macos"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

# https://aur.archlinux.org/cgit/aur.git/tree/PKGBUILD?h=sdl2-static
cmake(
    name = "sdl2_static",
    generate_args = [
        "-GNinja",
        "-DCMAKE_BUILD_TYPE=Release",  # always compile for release
        "-DCMAKE_INSTALL_LIBDIR=lib",
        "-DSDL_STATIC=ON",
        "-DSDL_STATIC_LIB=ON",
        "-DSDL_DLOPEN=ON",
        "-DARTS=OFF",
        "-DESD=OFF",
        "-DNAS=OFF",
        "-DHIDAPI=ON",
        "-DRPATH=OFF",
    ] + select({
        "@envpool//:windows": [],
        "//conditions:default": [
            "-DALSA=ON",
            "-DPULSEAUDIO_SHARED=ON",
            "-DVIDEO_WAYLAND=ON",
            "-DCLOCK_GETTIME=ON",
            "-DJACK_SHARED=ON",
            "-DSDL_STATIC_PIC=ON",
        ],
    }),
    lib_source = ":srcs",
    out_include_dir = "include",
    out_static_libs = select({
        "@envpool//:windows": ["SDL2-static.lib"],
        "//conditions:default": ["libSDL2.a"],
    }),
    visibility = ["//visibility:public"],
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
    deps = [":sdl2_static"],
)
