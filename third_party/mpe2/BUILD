# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

exports_files(["mpe2.BUILD"])

py_binary(
    name = "generate_metadata",
    srcs = ["generate_metadata.py"],
)

genrule(
    name = "metadata",
    srcs = [
        "@mpe2_upstream//:source",
        "@mpe2_upstream//:mpe2/__init__.py",
        "@mpe2_upstream//:mpe2/_mpe_utils/secrcode.ttf",
    ],
    outs = [
        "tasks.h",
        "font.h",
    ],
    cmd = "$(execpath :generate_metadata) --source $(execpath @mpe2_upstream//:mpe2/__init__.py) --output $(execpath tasks.h) --font-output $(execpath font.h)",
    tools = [":generate_metadata"],
)

genrule(
    name = "pcg_header",
    srcs = ["@mpe2_pcg64//file"],
    outs = ["pcg64.h"],
    cmd = "cp $< $@",
    cmd_bat = "copy /Y $< $@ >NUL",
)

cc_library(
    name = "metadata_headers",
    hdrs = [
        ":metadata",
        ":pcg_header",
    ],
)
