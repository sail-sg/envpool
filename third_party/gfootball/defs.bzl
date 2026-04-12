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

"""Macros for EnvPool's gfootball-generated files."""

load("//third_party:common.bzl", "copy_files_to_directory")

def envpool_gfootball_generated_files(name):
    copy_files_to_directory(
        name = "gen_gfootball_assets_data",
        srcs = ["@google_research_football//:engine_headless_data"],
        out = "assets/data",
        strip_prefix = "third_party/gfootball_engine/data/",
    )

    copy_files_to_directory(
        name = "gen_gfootball_assets_fonts",
        srcs = ["@google_research_football//:engine_fonts"],
        out = "assets/fonts",
        strip_prefix = "third_party/fonts/",
    )

    native.genrule(
        name = "gen_gfootball_scenarios_inc",
        srcs = ["@google_research_football//:gfootball_python"],
        outs = ["gfootball_scenarios.inc"],
        cmd = "$(location //third_party/gfootball:generate_scenarios) $(locations @google_research_football//:gfootball_python) > $@",
        tools = ["//third_party/gfootball:generate_scenarios"],
    )

    native.filegroup(
        name = name,
        srcs = [
            ":gen_gfootball_assets_data",
            ":gen_gfootball_assets_fonts",
            ":gen_gfootball_scenarios_inc",
        ],
    )
