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

genrule(
    name = "sdl_h",
    outs = ["SDL.h"],
    cmd = "cat >$@ <<'EOF'\n#include \"SDL2/SDL.h\"\nEOF",
)

genrule(
    name = "sdl2_rotozoom_prefixed_h",
    srcs = ["SDL2_rotozoom.h"],
    outs = ["SDL2/SDL2_rotozoom.h"],
    cmd = "mkdir -p $(@D) && cp $(location SDL2_rotozoom.h) $@",
)

cc_library(
    name = "sdl2_gfx",
    srcs = ["SDL2_rotozoom.c"],
    hdrs = [
        "SDL2_rotozoom.h",
        ":sdl_h",
        ":sdl2_rotozoom_prefixed_h",
    ],
    includes = ["."],
    deps = ["@sdl2"],
)
