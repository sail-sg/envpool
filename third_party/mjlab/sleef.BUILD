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
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE.txt"])

filegroup(
    name = "source",
    srcs = glob(["**"]),
)

cmake(
    name = "build",
    cache_entries = {
        "BUILD_SHARED_LIBS": "OFF",
        "CMAKE_INSTALL_LIBDIR": "lib",
        "CMAKE_POLICY_VERSION_MINIMUM": "3.5",
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "SLEEF_BUILD_GNUABI_LIBS": "OFF",
        "SLEEF_BUILD_SCALAR_LIB": "ON",
        "SLEEF_BUILD_TESTS": "OFF",
        "SLEEF_DISABLE_OPENMP": "ON",
    },
    # CMake detects its native Windows architecture from this variable, which
    # the hermetic action otherwise omits. A blank processor produces no
    # architecture-specific SLEEF declarations.
    env = select({
        "@envpool//:windows": {"PROCESSOR_ARCHITECTURE": "AMD64"},
        "//conditions:default": {},
    }),
    lib_source = ":source",
    out_static_libs = select({
        "@envpool//:windows": ["sleef.lib"],
        "//conditions:default": ["libsleef.a"],
    }),
)

cc_library(
    name = "sleef",
    defines = ["SLEEF_STATIC_LIBS=1"],
    deps = [":build"],
)
