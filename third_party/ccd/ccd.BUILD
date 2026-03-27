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

load("@envpool//third_party:common.bzl", "template_rule")
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

template_rule(
    name = "ccd_config",
    src = "src/ccd/config.h.cmake.in",
    out = "src/ccd/config.h",
    substitutions = {
        "#cmakedefine CCD_SINGLE": "",
        "#cmakedefine CCD_DOUBLE": "#define CCD_DOUBLE",
    },
)

cc_library(
    name = "ccd",
    srcs = glob([
        "src/*.c",
        "src/*.h",
    ]) + [":ccd_config"],
    hdrs = glob(["src/ccd/*.h"]),
    copts = ["-DCCD_STATIC_DEFINE"],
    includes = ["src"],
)
