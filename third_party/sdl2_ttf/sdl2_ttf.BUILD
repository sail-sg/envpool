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
    cmd_bat = "@echo #include \"SDL2/SDL.h\" > $@",
)

genrule(
    name = "sdl_cpuinfo_h",
    outs = ["SDL_cpuinfo.h"],
    cmd = "cat >$@ <<'EOF'\n#include \"SDL2/SDL_cpuinfo.h\"\nEOF",
    cmd_bat = "@echo #include \"SDL2/SDL_cpuinfo.h\" > $@",
)

genrule(
    name = "sdl_endian_h",
    outs = ["SDL_endian.h"],
    cmd = "cat >$@ <<'EOF'\n#include \"SDL2/SDL_endian.h\"\nEOF",
    cmd_bat = "@echo #include \"SDL2/SDL_endian.h\" > $@",
)

genrule(
    name = "begin_code_h",
    outs = ["begin_code.h"],
    cmd = "cat >$@ <<'EOF'\n#include \"SDL2/begin_code.h\"\nEOF",
    cmd_bat = "@echo #include \"SDL2/begin_code.h\" > $@",
)

genrule(
    name = "close_code_h",
    outs = ["close_code.h"],
    cmd = "cat >$@ <<'EOF'\n#include \"SDL2/close_code.h\"\nEOF",
    cmd_bat = "@echo #include \"SDL2/close_code.h\" > $@",
)

genrule(
    name = "sdl2_sdl_ttf_h",
    srcs = ["SDL_ttf.h"],
    outs = ["SDL2/SDL_ttf.h"],
    cmd = "mkdir -p $(@D) && cp $(location SDL_ttf.h) $@",
    cmd_bat = "if not exist \"$(@D)\" mkdir \"$(@D)\" & copy /Y \"$(location SDL_ttf.h)\" \"$@\" >NUL",
    cmd_ps = "$ErrorActionPreference = 'Stop'; New-Item -ItemType Directory -Force -Path '$(@D)' | Out-Null; Copy-Item -LiteralPath '$(location SDL_ttf.h)' -Destination '$@' -Force",
)

cc_library(
    name = "sdl2_ttf",
    srcs = ["SDL_ttf.c"],
    hdrs = [
        "SDL_ttf.h",
        ":begin_code_h",
        ":close_code_h",
        ":sdl2_sdl_ttf_h",
        ":sdl_cpuinfo_h",
        ":sdl_endian_h",
        ":sdl_h",
    ],
    includes = ["."],
    deps = [
        "@freetype",
        "@sdl2",
    ],
)
