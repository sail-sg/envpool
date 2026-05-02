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

load("@python_versions//3.12:defs.bzl", py_binary_312 = "py_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//envpool:myosuite_oracle_requirements.bzl", "oracle_requirement")
load("//third_party/myosuite:defs.bzl", "myosuite_native_assets")

exports_files([
    "oracle_requirements.txt",
    "myosuite_source.BUILD",
    "simhive_source.BUILD",
    "generate_render_sample.py",
])

py_binary_312(
    name = "generate_native_assets",
    srcs = [
        "generate_native_assets.py",
        "generate_reference_data.py",
        "generate_task_metadata.py",
        "generate_task_registry.py",
    ],
    deps = [
        oracle_requirement("click"),
        oracle_requirement("dm-control"),
        oracle_requirement("flatten-dict"),
        oracle_requirement("gitpython"),
        oracle_requirement("gymnasium"),
        oracle_requirement("h5py"),
        oracle_requirement("mujoco"),
        oracle_requirement("numpy"),
        oracle_requirement("packaging"),
        oracle_requirement("pillow"),
        oracle_requirement("pink-noise-rl"),
        oracle_requirement("sk-video"),
        oracle_requirement("termcolor"),
    ],
)

py_binary_312(
    name = "generate_runtime_assets",
    srcs = ["generate_runtime_assets.py"],
    visibility = ["//visibility:public"],
    deps = [
        oracle_requirement("mujoco"),
        oracle_requirement("typing_extensions"),
    ],
)

myosuite_native_assets(
    name = "gen_myosuite_native_assets",
    furniture_srcs = ["@myosuite_furniture_sim//:source"],
    generator = ":generate_native_assets",
    mpl_srcs = ["@myosuite_mpl_sim//:source"],
    myo_srcs = ["@myosuite_myo_sim//:source"],
    myosuite_srcs = ["@myosuite_source//:source"],
    object_srcs = ["@myosuite_object_sim//:source"],
    ycb_srcs = ["@myosuite_ycb_sim//:source"],
)

filegroup(
    name = "myosuite_native_headers",
    srcs = [":gen_myosuite_native_assets"],
    output_group = "headers",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "myosuite_generated_tasks_py",
    srcs = [":gen_myosuite_native_assets"],
    output_group = "python",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "myosuite_generated_json",
    srcs = [":gen_myosuite_native_assets"],
    output_group = "json",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "myosuite_tasks",
    hdrs = [":myosuite_native_headers"],
    visibility = ["//visibility:public"],
)
