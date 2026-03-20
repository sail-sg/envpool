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

load("@envpool//third_party:common.bzl", "template_rule")
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "irregular_files",
    hdrs = glob([
        "src/**/*.def",
        "src/**/*.ins",
    ]),
)

template_rule(
    name = "ale_version",
    src = "src/version.hpp.in",
    out = "version.hpp",
    substitutions = {
        "@ALE_VERSION@": "0.7.5",
        "@ALE_VERSION_MAJOR@": "0",
        "@ALE_VERSION_MINOR@": "7",
        "@ALE_VERSION_PATCH@": "5",
        "@ALE_VERSION_GIT_SHA@": "978d2ce25665338ba71e45d32fff853b17c15f2e",
    },
)

cc_library(
    name = "ale_interface",
    srcs = glob(
        [
            "src/**/*.h",
            "src/**/*.hpp",
            "src/**/*.hxx",
            "src/**/*.c",
            "src/**/*.cpp",
            "src/**/*.cxx",
        ],
        exclude = [
            "src/python/*",
        ],
    ) + [
        ":ale_version",
    ],
    hdrs = ["src/ale_interface.hpp"],
    copts = [
        "-include",
        "cstdint",
    ],
    includes = [
        "src",
        "src/common",
        "src/emucore",
        "src/environment",
        "src/games",
        "src/games/supported",
    ],
    linkopts = [
        "-ldl",
    ],
    linkstatic = 0,
    deps = [
        ":irregular_files",
        "@zlib",
    ],
)
