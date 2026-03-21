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

# Description:
#   NASM is a portable assembler in the Intel/Microsoft tradition.

load("@rules_cc//cc:defs.bzl", "cc_binary")

licenses(["notice"])  # BSD 2-clause

exports_files([
    "LICENSE",
    "config.h",
])

genrule(
    name = "config_h",
    srcs = ["@//third_party/nasm:config.h"],
    outs = ["config/config.h"],
    cmd = "cp $(location @//third_party/nasm:config.h) $@",
)

cc_binary(
    name = "nasm",
    srcs = glob([
        "asm/*.c",
        "asm/*.h",
        "autoconf/*.h",
        "common/*.c",
        "config/*.h",
        "include/*.h",
        "macros/*.c",
        "nasmlib/*.c",
        "nasmlib/*.h",
        "output/*.c",
        "output/*.h",
        "stdlib/*.c",
        "x86/*.c",
        "x86/*.h",
        "version.h",
        "zlib/*.c",
        "zlib/*.h",
    ]) + [":config_h"] + select({
        ":windows": ["config/msvc.h"],
        "//conditions:default": [],
    }),
    copts = select({
        ":windows": [],
        "//conditions:default": [
            "-w",
            "-std=c17",
            "-U__STRICT_ANSI__",
            "-fno-common",
            "-fwrapv",
        ],
    }),
    defines = select({
        ":windows": [],
        "//conditions:default": [
            "HAVE_CONFIG_H",
        ],
    }),
    includes = [
        "asm",
        "autoconf",
        "config",
        "include",
        "output",
        "x86",
        "zlib",
    ],
    visibility = ["@libjpeg_turbo//:__pkg__"],
)

config_setting(
    name = "windows",
    values = {
        "cpu": "x64_windows",
    },
)
