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

load("@rules_cc//cc:cc_library.bzl", "cc_library")

exports_files([
    "oracle_requirements.txt",
    "myosuite_source.BUILD",
    "myosuite_reference_data.h",
    "myosuite_tasks.h",
    "myosuite_tasks.json",
    "simhive_source.BUILD",
])

cc_library(
    name = "myosuite_tasks",
    hdrs = [
        "myosuite_reference_data.h",
        "myosuite_task_metadata.h",
        "myosuite_tasks.h",
    ],
    visibility = ["//visibility:public"],
)
