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

"""Shared Bazel test macros for MyoSuite."""

load("@rules_python//python:defs.bzl", "py_test")

def myosuite_sharded_py_test(
        name,
        src,
        env_prefix,
        shard_count,
        data = None,
        deps = None):
    """Creates explicit MyoSuite task shards.

    Args:
      name: Public test_suite target name.
      src: Python test source file.
      env_prefix: Prefix for TOTAL_SHARDS and SHARD_INDEX environment keys.
      shard_count: Number of task shards to create.
      data: Optional runtime data labels for every shard.
      deps: Optional dependency labels for every shard.
    """
    if data == None:
        data = []
    if deps == None:
        deps = []
    shard_targets = []
    for shard_index in range(shard_count):
        shard_name = "%s_shard_%d" % (name, shard_index)
        py_test(
            name = shard_name,
            size = "enormous",
            srcs = [src],
            main = src,
            data = data,
            env = {
                "%s_SHARD_INDEX" % env_prefix: str(shard_index),
                "%s_TOTAL_SHARDS" % env_prefix: str(shard_count),
            },
            imports = ["../.."],
            tags = ["exclusive"],
            deps = deps,
        )
        shard_targets.append(":" + shard_name)
    native.test_suite(
        name = name,
        tests = shard_targets,
    )
