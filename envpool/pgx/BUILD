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

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//envpool:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pgx_env",
    hdrs = [
        "animal_shogi.h",
        "backgammon.h",
        "board_games.h",
        "card_games.h",
        "chess_games.h",
        "go.h",
        "play2048.h",
        "shogi.h",
        "sparrow_mahjong.h",
    ],
    deps = [
        "//envpool/core:async_envpool",
        "//envpool/core:env",
    ],
)

pybind_extension(
    name = "pgx_envpool",
    srcs = ["pgx.cc"],
    deps = [
        ":pgx_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "pgx",
    srcs = ["__init__.py"],
    data = [":pgx_envpool"],
    imports = ["../.."],
    deps = ["//envpool/python:api"],
)

py_library(
    name = "pgx_registration",
    srcs = ["registration.py"],
    imports = ["../.."],
    deps = ["//envpool:registration"],
)

py_test(
    name = "pgx_test",
    srcs = ["pgx_test.py"],
    data = [":pgx_envpool"],
    imports = ["../.."],
    deps = [
        ":pgx",
        ":pgx_registration",
        requirement("absl-py"),
        requirement("gymnasium"),
        requirement("numpy"),
    ],
)

py_test(
    name = "pgx_align_test",
    timeout = "long",
    srcs = ["pgx_align_test.py"],
    data = [":pgx_envpool"],
    imports = ["../.."],
    deps = [
        ":pgx",
        ":pgx_registration",
        requirement("absl-py"),
        requirement("jax"),
        requirement("numpy"),
        requirement("pgx"),
    ],
)

py_test(
    name = "pgx_deterministic_test",
    srcs = ["pgx_deterministic_test.py"],
    data = [":pgx_envpool"],
    imports = ["../.."],
    deps = [
        ":pgx",
        ":pgx_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "pgx_render_test",
    srcs = ["pgx_render_test.py"],
    data = [":pgx_envpool"],
    imports = ["../.."],
    deps = [
        ":pgx",
        ":pgx_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)
