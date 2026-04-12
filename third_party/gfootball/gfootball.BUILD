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

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "darwin",
    constraint_values = ["@platforms//os:macos"],
)

filegroup(
    name = "gfootball_python",
    srcs = glob(["gfootball/**/*.py"]),
)

filegroup(
    name = "engine_data",
    srcs = glob(["third_party/gfootball_engine/data/**"]),
)

filegroup(
    name = "engine_fonts",
    srcs = glob(["third_party/fonts/**"]),
)

cc_library(
    name = "gfootball_engine",
    srcs = glob(
        ["third_party/gfootball_engine/src/**/*.cpp"],
        exclude = [
            "third_party/gfootball_engine/src/client.cpp",
            "third_party/gfootball_engine/src/cmake/backtrace.cpp",
        ],
    ) + select({
        "@envpool//:windows": [],
        "//conditions:default": [
            "third_party/gfootball_engine/src/cmake/backtrace.cpp",
        ],
    }),
    hdrs = glob([
        "third_party/gfootball_engine/src/**/*.h",
        "third_party/gfootball_engine/src/**/*.hpp",
    ]),
    includes = [
        "third_party/gfootball_engine/src",
        "third_party/gfootball_engine/src/cmake",
    ],
    linkopts = select({
        ":darwin": [
            "-framework OpenGL",
        ],
        "@envpool//:windows": [],
        "//conditions:default": [
            "-ldl",
            "-lpthread",
            "-lEGL",
            "-lOpenGL",
        ],
    }),
    deps = [
        "@boost//:algorithm",
        "@boost//:bind",
        "@boost//:circular_buffer",
        "@boost//:filesystem",
        "@boost//:intrusive",
        "@boost//:random",
        "@boost//:signals2",
        "@boost//:smart_ptr",
        "@boost//:system",
        "@boost//:thread",
        "@sdl2",
        "@sdl2_gfx",
        "@sdl2_ttf",
    ],
)
