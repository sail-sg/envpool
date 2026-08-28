# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

alias(
    name = "maze",
    actual = "//labmaze/cc:random_maze",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "textures",
    srcs = glob([
        "labmaze/assets/style_01/*.png",
        "labmaze/assets/sky_03/*.png",
    ]),
    visibility = ["//visibility:public"],
)

exports_files([
    "labmaze/assets/__init__.py",
    "LICENSE",
])
