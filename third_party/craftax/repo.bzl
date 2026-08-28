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

"""Pinned Craftax source, oracle and texture assets."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def craftax_archive():
    """Fetch the complete official v1.6.1 source without runtime Python deps."""
    http_archive(
        name = "craftax_upstream",
        build_file = "//third_party/craftax:craftax.BUILD",
        sha256 = "6a9939698ea3279cc830ae8d106e33de1b0f21c95e030debbaef0fd891079541",
        strip_prefix = "Craftax-c3c2e0d038c4e641f9481320c158f457f30c28f3",
        type = "tar.gz",
        urls = ["https://codeload.github.com/MichaelTMatthews/Craftax/tar.gz/c3c2e0d038c4e641f9481320c158f457f30c28f3"],
    )
