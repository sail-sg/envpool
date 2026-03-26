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

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

template_rule(
    name = "vizdoom_version",
    src = "src/lib/ViZDoomVersion.h.in",
    out = "ViZDoomVersion.h",
    substitutions = {
        "@ViZDoom_VERSION_ID@": "1300",
        "@ViZDoom_VERSION_STR@": "1.3.0",
    },
)

cc_library(
    name = "vizdoom_lib",
    srcs = glob([
        "include/*.h",
        "src/lib/*.h",
        "src/lib/*.cpp",
        "src/lib/boost/**/*.hpp",
    ]) + [":vizdoom_version"],
    hdrs = glob([
        "include/*.h",
        "src/vizdoom/src/**/*.h",
        "src/vizdoom/src/**/*.hpp",
    ]),
    includes = [
        "include",
        "src/lib",
    ],
    linkopts = select({
        ":windows": [],
        "//conditions:default": ["-ldl"],
    }),
    deps = [
        "@boost//:asio",
        "@boost//:filesystem",
        "@boost//:interprocess",
        "@boost//:iostreams",
        "@boost//:process",
        "@boost//:random",
        "@boost//:thread",
    ],
)

filegroup(
    name = "vizdoom_maps",
    srcs = glob(["scenarios/*"]),
)
