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
"""Bazel helpers for MyoSuite render validation shards."""

load("@rules_python//python:defs.bzl", "py_test")
load("//envpool:requirements.bzl", "requirement")

def myosuite_render_shard_tests(shard_count):
    """Declare one MyoSuite render test target per shard."""
    for shard_index in range(shard_count):
        py_test(
            name = "myosuite_render_shard_%d" % shard_index,
            size = "large",
            srcs = ["myosuite/myosuite_render_test.py"],
            args = [
                "--myosuite_render_shard_index=%d" % shard_index,
                "--myosuite_render_shard_count=%d" % shard_count,
                "--myosuite_render_include_doc_cases=false",
            ],
            imports = ["../.."],
            main = "myosuite/myosuite_render_test.py",
            tags = ["exclusive"],
            deps = [
                ":myosuite_render_utils",
                requirement("absl-py"),
                requirement("numpy"),
            ],
        )
