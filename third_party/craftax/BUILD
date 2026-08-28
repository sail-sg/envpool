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
load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

exports_files(["craftax.BUILD"])

py_binary(
    name = "generate_metadata",
    srcs = ["generate_metadata.py"],
)

py_binary(
    name = "generate_assets",
    srcs = ["generate_assets.py"],
)

genrule(
    name = "embedded_textures",
    srcs = [
        "@craftax_upstream//:textures",
        "@craftax_upstream//:craftax/craftax/constants.py",
        "@craftax_upstream//:craftax/craftax_classic/constants.py",
    ],
    outs = ["textures.cc"],
    cmd = "$(execpath :generate_assets) --classic $(execpath @craftax_upstream//:craftax/craftax_classic/constants.py) --full $(execpath @craftax_upstream//:craftax/craftax/constants.py) --output $@",
    tools = [":generate_assets"],
)

genrule(
    name = "generate_constants",
    srcs = [
        "@craftax_upstream//:craftax/craftax/constants.py",
        "@craftax_upstream//:craftax/craftax_classic/constants.py",
        "@craftax_upstream//:craftax/craftax/world_gen/world_gen_configs.py",
    ],
    outs = [
        "constants.h",
        "info.h",
    ],
    cmd = "$(execpath :generate_metadata) --classic $(execpath @craftax_upstream//:craftax/craftax_classic/constants.py) --full $(execpath @craftax_upstream//:craftax/craftax/constants.py) --world $(execpath @craftax_upstream//:craftax/craftax/world_gen/world_gen_configs.py) --output $(execpath constants.h) --info-output $(execpath info.h)",
    tools = [":generate_metadata"],
)

cc_library(
    name = "constants",
    hdrs = [":generate_constants"],
)

py_binary(
    name = "render_compare",
    testonly = True,
    srcs = ["render_compare.py"],
    deps = [
        "//envpool/craftax:craftax_registration",
        "//envpool/craftax:oracle",
    ],
)
