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

load("@envpool//envpool:requirements.bzl", "requirement")
load("@envpool//third_party/craftax:oracle_requirements.bzl", "craftax_oracle_requirement")
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "oracle",
    testonly = True,
    srcs = glob(["src/xminigrid/**/*.py"]),
    imports = ["src"],
    deps = [
        craftax_oracle_requirement("flax"),
        craftax_oracle_requirement("jax"),
        requirement("tqdm"),
    ],
)
