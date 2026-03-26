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
    src = "src/ale/version.hpp.in",
    out = "version.hpp",
    substitutions = {
        "@ALE_VERSION@": "0.11.2",
        "@ALE_VERSION_MAJOR@": "0",
        "@ALE_VERSION_MINOR@": "11",
        "@ALE_VERSION_PATCH@": "2",
        "@ALE_VERSION_GIT_SHA@": "ecc113829d571348adc1a299fcf1321238dd684e",
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
            "src/ale/python/**",
        ],
    ),
    hdrs = [
        "src/ale/ale_interface.hpp",
        ":ale_version",
    ],
    copts = [
        "-include",
        "cstdint",
    ],
    includes = [
        "src",
        "src/ale",
    ],
    linkopts = select({
        "@envpool//:windows": [],
        "//conditions:default": [
            "-ldl",
        ],
    }),
    linkstatic = 0,
    deps = [
        ":irregular_files",
        "@zlib",
    ],
)
