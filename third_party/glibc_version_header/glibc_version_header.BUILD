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

exports_files(
    ["noop.h"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
)

filegroup(
    name = "glibc_2_17",
    srcs = select({
        ":linux_x86_64": ["force_link_glibc_2.17.h"],
        "//conditions:default": ["@envpool//third_party/glibc_version_header:noop.h"],
    }),
    visibility = ["//visibility:public"],
)
