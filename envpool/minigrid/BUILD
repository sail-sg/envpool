# Copyright 2023 Garena Online Private Limited
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

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//envpool:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "minigrid_env",
    srcs = [
        "impl/babyai_core.cc",
        "impl/babyai_env.cc",
        "impl/babyai_goto_tasks.cc",
        "impl/babyai_instr.cc",
        "impl/babyai_open_tasks.cc",
        "impl/babyai_pickup_tasks.cc",
        "impl/babyai_tasks.cc",
        "impl/babyai_unlock_tasks.cc",
        "impl/minigrid_core.cc",
        "impl/minigrid_env.cc",
        "impl/minigrid_render.cc",
        "impl/minigrid_room_tasks.cc",
        "impl/minigrid_task_factory.cc",
        "impl/minigrid_tasks.cc",
        "impl/minigrid_wfc_tasks.cc",
    ],
    hdrs = [
        "impl/babyai_core.h",
        "impl/babyai_env.h",
        "impl/babyai_task_factory.h",
        "impl/minigrid_env.h",
        "impl/minigrid_task_utils.h",
        "impl/minigrid_task_factory.h",
        "impl/utils.h",
        "minigrid.h",
    ],
    deps = [
        "//envpool/core:async_envpool",
        "//envpool/core:logging",
        "@opencv",
    ],
)

pybind_extension(
    name = "minigrid_envpool",
    srcs = [
        "minigrid.cc",
        "minigrid_bindings.cc",
    ],
    deps = [
        ":minigrid_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "minigrid",
    srcs = ["__init__.py"],
    imports = ["../.."],
    data = [":minigrid_envpool"],
    deps = [
        "//envpool/python:api",
        requirement("numpy"),
    ],
)

py_library(
    name = "minigrid_registration",
    srcs = ["registration.py"],
    imports = ["../.."],
    deps = [
        "//envpool:registration",
    ],
)

py_test(
    name = "babyai_align_test",
    size = "enormous",
    srcs = [
        "babyai_align_test.py",
        "babyai_test_utils.py",
    ],
    imports = ["../.."],
    deps = [
        ":minigrid",
        ":minigrid_registration",
        requirement("absl-py"),
        requirement("gymnasium"),
        requirement("numpy"),
        requirement("minigrid"),
    ],
)

py_test(
    name = "babyai_deterministic_test",
    size = "enormous",
    srcs = [
        "babyai_deterministic_test.py",
        "babyai_test_utils.py",
    ],
    imports = ["../.."],
    deps = [
        ":minigrid",
        ":minigrid_registration",
        requirement("absl-py"),
        requirement("gymnasium"),
        requirement("numpy"),
        requirement("minigrid"),
    ],
)

py_test(
    name = "babyai_render_test",
    size = "enormous",
    srcs = [
        "babyai_render_test.py",
        "babyai_test_utils.py",
    ],
    imports = ["../.."],
    deps = [
        ":minigrid",
        ":minigrid_registration",
        requirement("absl-py"),
        requirement("gymnasium"),
        requirement("numpy"),
        requirement("minigrid"),
    ],
)

py_test(
    name = "minigrid_align_test",
    size = "enormous",
    srcs = ["minigrid_align_test.py"],
    imports = ["../.."],
    deps = [
        ":minigrid",
        ":minigrid_registration",
        requirement("absl-py"),
        requirement("gymnasium"),
        requirement("numpy"),
        requirement("minigrid"),
    ],
)

py_test(
    name = "minigrid_deterministic_test",
    size = "enormous",
    srcs = ["minigrid_deterministic_test.py"],
    imports = ["../.."],
    deps = [
        ":minigrid",
        ":minigrid_registration",
        requirement("absl-py"),
        requirement("gymnasium"),
        requirement("numpy"),
    ],
)

py_test(
    name = "minigrid_render_test",
    size = "enormous",
    srcs = ["minigrid_render_test.py"],
    imports = ["../.."],
    deps = [
        ":minigrid",
        ":minigrid_registration",
        requirement("absl-py"),
        requirement("gymnasium"),
        requirement("numpy"),
        requirement("minigrid"),
    ],
)
