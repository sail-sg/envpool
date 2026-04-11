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
load("//third_party:common.bzl", "copy_to_directory")

package(default_visibility = ["//visibility:public"])

copy_to_directory(
    name = "gen_gfootball_assets_data",
    srcs = ["@google_research_football//:engine_data"],
    out = "assets/data",
    strip_prefix = "third_party/gfootball_engine/data/",
)

copy_to_directory(
    name = "gen_gfootball_assets_fonts",
    srcs = ["@google_research_football//:engine_fonts"],
    out = "assets/fonts",
    strip_prefix = "third_party/fonts/",
)

genrule(
    name = "gen_gfootball_scenarios_inc",
    srcs = ["@google_research_football//:gfootball_python"],
    outs = ["gfootball_scenarios.inc"],
    cmd = "$(location //third_party/gfootball:generate_scenarios) $(locations @google_research_football//:gfootball_python) > $@",
    tools = ["//third_party/gfootball:generate_scenarios"],
)

cc_library(
    name = "gfootball_env",
    hdrs = [
        "gfootball_common.h",
        "gfootball_env.h",
        ":gen_gfootball_scenarios_inc",
    ],
    data = [
        ":gen_gfootball_assets_data",
        ":gen_gfootball_assets_fonts",
    ],
    deps = [
        "//envpool/core:async_envpool",
        "//envpool/utils:image_process",
        "@google_research_football//:gfootball_engine",
    ],
)

pybind_extension(
    name = "gfootball_envpool",
    srcs = ["gfootball_envpool.cc"],
    deps = [
        ":gfootball_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "gfootball",
    srcs = ["__init__.py"],
    data = [
        ":gen_gfootball_assets_data",
        ":gen_gfootball_assets_fonts",
        ":gfootball_envpool",
    ],
    imports = ["../.."],
    deps = [
        "//envpool/python:api",
        requirement("numpy"),
    ],
)

py_library(
    name = "gfootball_registration",
    srcs = ["registration.py"],
    imports = ["../.."],
    deps = ["//envpool:registration"],
)

py_test(
    name = "gfootball_align_test",
    size = "enormous",
    srcs = [
        "gfootball_align_test.py",
        "gfootball_oracle_util.py",
    ],
    imports = ["../.."],
    deps = [
        ":gfootball",
        ":gfootball_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "gfootball_render_test",
    size = "enormous",
    srcs = [
        "gfootball_oracle_util.py",
        "gfootball_render_test.py",
    ],
    imports = ["../.."],
    deps = [
        ":gfootball",
        ":gfootball_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "gfootball_deterministic_test",
    size = "enormous",
    srcs = [
        "gfootball_deterministic_test.py",
        "gfootball_oracle_util.py",
    ],
    imports = ["../.."],
    deps = [
        ":gfootball",
        ":gfootball_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)
