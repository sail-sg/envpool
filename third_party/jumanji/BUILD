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
load("@rules_python//python:defs.bzl", "py_binary")

exports_files([
    "matplotlib_311_autoscale.patch",
    "oracle_requirements.txt",
])

py_binary(
    name = "generate_pacman_maze",
    srcs = ["generate_pacman_maze.py"],
)

genrule(
    name = "pacman_maze_header",
    srcs = ["@jumanji_pacman_constants//file"],
    outs = ["pacman_maze.h"],
    cmd = "$(execpath :generate_pacman_maze) $(execpath @jumanji_pacman_constants//file) $@",
    tools = [":generate_pacman_maze"],
)

cc_library(
    name = "pacman_maze",
    hdrs = [":pacman_maze_header"],
    visibility = ["//visibility:public"],
)
