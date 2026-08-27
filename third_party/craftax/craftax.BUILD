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

load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

exports_files([
    "LICENSE",
    "craftax/craftax/constants.py",
    "craftax/craftax_classic/constants.py",
    "craftax/craftax_env.py",
    "craftax/craftax/world_gen/world_gen_configs.py",
])

filegroup(
    name = "textures",
    srcs = glob([
        "craftax/craftax/assets/*.png",
        "craftax/craftax_classic/assets/*.png",
    ]),
)

py_library(
    name = "oracle",
    testonly = True,
    srcs = glob(
        ["craftax/**/*.py"],
        exclude = ["**/play_craftax*.py"],
    ),
    data = [":textures"],
    imports = ["."],
)
