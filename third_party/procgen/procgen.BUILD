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

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "procgen_assets",
    srcs = [
        "data/assets/kenney",
        "data/assets/kenney-abstract",
        "data/assets/misc_assets",
        "data/assets/platform_backgrounds",
        "data/assets/platform_backgrounds_2",
        "data/assets/platformer",
        "data/assets/space_backgrounds",
        "data/assets/topdown_backgrounds",
        "data/assets/water_backgrounds",
    ],
)

cc_library(
    name = "procgen",
    srcs = glob(["src/**/*.cpp"]) + glob(["src/*.h"]),
    hdrs = glob(["src/*.h"]),
    copts = [
        "-fpic",
    ],
    strip_include_prefix = "src",
    deps = [
        "@gym3_libenv//:gym3_libenv_header",
        "@qt//:qt_core",
        "@qt//:qt_gui",
    ],
    alwayslink = True,
)
