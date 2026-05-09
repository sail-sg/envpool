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
load("//third_party/jumanji:oracle_requirements.bzl", "jumanji_oracle_requirement")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "jumanji_env",
    hdrs = [
        "bin_pack_env.h",
        "cleaner_env.h",
        "connector_env.h",
        "cvrp_env.h",
        "flat_pack_env.h",
        "game2048_env.h",
        "graph_coloring_env.h",
        "job_shop_env.h",
        "jumanji_env.h",
        "knapsack_env.h",
        "level_based_foraging_env.h",
        "maze_env.h",
        "minesweeper_env.h",
        "mmst_env.h",
        "multi_cvrp_env.h",
        "pac_man_env.h",
        "parse_utils.h",
        "render_utils.h",
        "robot_warehouse_env.h",
        "rubiks_cube_env.h",
        "search_and_rescue_env.h",
        "sliding_tile_puzzle_env.h",
        "snake_env.h",
        "sokoban_env.h",
        "sudoku_env.h",
        "tetris_env.h",
        "tsp_env.h",
    ],
    deps = [
        "//envpool/core:async_envpool",
    ],
)

pybind_extension(
    name = "jumanji_envpool",
    srcs = ["jumanji_envpool.cc"],
    deps = [
        ":jumanji_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "jumanji",
    srcs = glob(["_official_render/**/*.py"]) + [
        "__init__.py",
        "jumanji_official_render.py",
        "jumanji_render.py",
    ],
    data = glob([
        "_official_render/**/*.jpeg",
        "_official_render/**/*.png",
    ]) + [
        ":boxoban_unfiltered_test",
        ":jumanji_envpool",
    ],
    imports = ["../.."],
    deps = [
        "//envpool/python:api",
        "//envpool/python:envpool",
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("pillow"),
    ],
)

filegroup(
    name = "boxoban_unfiltered_test",
    srcs = ["assets/boxoban/unfiltered-test.npy"],
)

py_library(
    name = "jumanji_registration",
    srcs = ["registration.py"],
    imports = ["../.."],
    deps = [
        "//envpool:registration",
    ],
)

py_test(
    name = "jumanji_registry_test",
    size = "enormous",
    srcs = ["jumanji_registry_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("gymnasium"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_official_align_test",
    size = "enormous",
    srcs = ["jumanji_official_align_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
        jumanji_oracle_requirement("jax"),
        jumanji_oracle_requirement("jumanji"),
        jumanji_oracle_requirement("matplotlib"),
        jumanji_oracle_requirement("requests"),
    ],
)

py_test(
    name = "jumanji_bin_pack_test",
    size = "small",
    srcs = ["jumanji_bin_pack_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_cleaner_test",
    size = "small",
    srcs = ["jumanji_cleaner_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_sudoku_test",
    size = "small",
    srcs = ["jumanji_sudoku_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_connector_test",
    size = "small",
    srcs = ["jumanji_connector_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_tsp_test",
    size = "small",
    srcs = ["jumanji_tsp_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_cvrp_test",
    size = "small",
    srcs = ["jumanji_cvrp_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_flat_pack_test",
    size = "small",
    srcs = ["jumanji_flat_pack_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_game2048_test",
    size = "small",
    srcs = ["jumanji_game2048_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_graph_coloring_test",
    size = "small",
    srcs = ["jumanji_graph_coloring_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_job_shop_test",
    size = "small",
    srcs = ["jumanji_job_shop_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_knapsack_test",
    size = "small",
    srcs = ["jumanji_knapsack_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_level_based_foraging_test",
    size = "small",
    srcs = ["jumanji_level_based_foraging_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_maze_test",
    size = "small",
    srcs = ["jumanji_maze_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_minesweeper_test",
    size = "small",
    srcs = ["jumanji_minesweeper_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_mmst_test",
    size = "small",
    srcs = ["jumanji_mmst_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_multi_cvrp_test",
    size = "small",
    srcs = ["jumanji_multi_cvrp_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_pac_man_test",
    size = "small",
    srcs = ["jumanji_pac_man_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_robot_warehouse_test",
    size = "small",
    srcs = ["jumanji_robot_warehouse_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_rubiks_cube_test",
    size = "small",
    srcs = ["jumanji_rubiks_cube_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_search_and_rescue_test",
    size = "small",
    srcs = ["jumanji_search_and_rescue_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_sliding_tile_puzzle_test",
    size = "small",
    srcs = ["jumanji_sliding_tile_puzzle_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_snake_test",
    size = "small",
    srcs = ["jumanji_snake_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_sokoban_test",
    size = "large",
    srcs = ["jumanji_sokoban_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "jumanji_tetris_test",
    size = "small",
    srcs = ["jumanji_tetris_test.py"],
    imports = ["../.."],
    deps = [
        ":jumanji",
        ":jumanji_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)
