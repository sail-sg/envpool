# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@dml//:requirements.bzl", asset_requirement = "requirement")
load("@python_versions//3.12:defs.bzl", py_binary_312 = "py_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

exports_files([
    "labmaze.BUILD",
    "labmaze.patch",
    "asset_requirements.txt",
    "LICENSE.scipy",
])

py_binary(
    name = "generate_metadata",
    srcs = ["generate_metadata.py"],
)

genrule(
    name = "metadata",
    srcs = [
        "@mujoco_dmc_xml//:__init__.py",
        "@mujoco_dmc_xml//:locomotion_metadata_source",
        "@labmaze_source//:labmaze/assets/__init__.py",
    ],
    outs = [
        "metadata.h",
        "tasks.json",
    ],
    cmd = "$(location :generate_metadata) --source $(location @mujoco_dmc_xml//:__init__.py) --labmaze $(location @labmaze_source//:labmaze/assets/__init__.py) --header $(location metadata.h) --registry $(location tasks.json)",
    tools = [":generate_metadata"],
)

cc_library(
    name = "task_metadata",
    hdrs = ["metadata.h"],
)

py_binary_312(
    name = "generate_mocap",
    srcs = ["generate_mocap.py"],
    deps = [
        asset_requirement("h5py"),
        asset_requirement("numpy"),
    ],
)

genrule(
    name = "mocap",
    srcs = [
        "@mujoco_dmc_xml//:__init__.py",
        "@mujoco_dmc_xml//:locomotion_mocap_source",
        "@dmc_cmu_mocap_2019//file",
        "@dmc_cmu_mocap_2020//file",
    ],
    outs = [
        "mocap_2019.bin",
        "mocap_2020.bin",
        "oracle_2019.h5",
        "oracle_2020.h5",
        "mocap.json",
    ],
    cmd = "$(location :generate_mocap) --source $(location @mujoco_dmc_xml//:__init__.py) --cmu-2019 $(location @dmc_cmu_mocap_2019//file) --cmu-2020 $(location @dmc_cmu_mocap_2020//file) --output $(@D)",
    tools = [":generate_mocap"],
)

filegroup(
    name = "mocap_runtime",
    srcs = [
        "LICENSE.scipy",
        "NOTICE",
        "mocap.json",
        "mocap_2019.bin",
        "mocap_2020.bin",
        "@labmaze_source//:LICENSE",
    ],
)

filegroup(
    name = "mocap_oracle",
    srcs = [
        "oracle_2019.h5",
        "oracle_2020.h5",
    ],
)
