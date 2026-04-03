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

load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//envpool:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "gymnasium_robotics",
    srcs = ["__init__.py"],
    imports = ["../.."],
    deps = [
        "//envpool/python:data",
        "//envpool/python:glfw_context",
        "//envpool/python:protocol",
        requirement("dm-env"),
        requirement("gymnasium"),
        requirement("gymnasium-robotics"),
        requirement("numpy"),
    ],
)

py_library(
    name = "gymnasium_robotics_registration",
    srcs = ["registration.py"],
    imports = ["../.."],
    deps = [
        ":gymnasium_robotics",
        "//envpool:registration",
        requirement("gymnasium"),
        requirement("gymnasium-robotics"),
    ],
)

py_test(
    name = "gymnasium_robotics_test",
    size = "enormous",
    srcs = ["gymnasium_robotics_test.py"],
    imports = ["../.."],
    deps = [
        ":gymnasium_robotics",
        ":gymnasium_robotics_registration",
        "//envpool:registration",
        requirement("dm-env"),
        requirement("absl-py"),
        requirement("gymnasium"),
        requirement("gymnasium-robotics"),
        requirement("numpy"),
    ],
)
