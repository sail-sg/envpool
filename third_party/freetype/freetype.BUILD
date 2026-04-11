# Copyright 2026 Garena Online Private Limited
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

load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "freetype_static",
    generate_args = [
        "-GNinja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DFT_DISABLE_ZLIB=TRUE",
        "-DFT_DISABLE_BZIP2=TRUE",
        "-DFT_DISABLE_PNG=TRUE",
        "-DFT_DISABLE_HARFBUZZ=TRUE",
        "-DFT_DISABLE_BROTLI=TRUE",
    ],
    lib_source = ":srcs",
    out_include_dir = "include/freetype2",
    out_static_libs = select({
        "@envpool//:windows": ["freetype.lib"],
        "//conditions:default": ["libfreetype.a"],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "freetype",
    visibility = ["//visibility:public"],
    deps = [":freetype_static"],
)
