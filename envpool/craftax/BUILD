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

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("//envpool:requirements.bzl", "requirement")
load("//third_party/craftax:oracle_requirements.bzl", "craftax_oracle_requirement")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "game",
    srcs = [
        "classic.cc",
        "full.cc",
        "full_actions.cc",
        "full_mobs.cc",
        "game.cc",
        "observation.cc",
        "renderer.cc",
        "world.cc",
        "world_full.cc",
        "//third_party/craftax:embedded_textures",
    ],
    hdrs = [
        "game.h",
        "noise.h",
        "random.h",
        "renderer.h",
        "state_io.h",
    ],
    copts = select({
        "//:windows": ["/fp:strict"],
        "//conditions:default": ["-ffp-contract=off"],
    }),
    deps = [
        "//third_party/craftax:constants",
        "@lodepng",
    ],
)

cc_library(
    name = "craftax_env",
    hdrs = ["craftax.h"],
    deps = [
        ":game",
        "//envpool/core:async_envpool",
        "//envpool/core:env",
    ],
)

pybind_extension(
    name = "craftax_envpool",
    srcs = ["craftax.cc"],
    deps = [
        ":craftax_env",
        "//envpool/core:py_envpool",
    ],
)

pybind_extension(
    name = "craftax_debug",
    testonly = True,
    srcs = ["craftax_debug.cc"],
    deps = [":game"],
)

py_library(
    name = "craftax",
    srcs = ["__init__.py"],
    data = [
        "LICENSE",
        ":craftax_envpool",
    ],
    imports = ["../.."],
    deps = ["//envpool/python:api"],
)

py_library(
    name = "craftax_registration",
    srcs = [
        "_registry.py",
        "registration.py",
    ],
    imports = ["../.."],
    deps = ["//envpool:registration"],
)

py_test(
    name = "craftax_test",
    timeout = "long",
    srcs = ["craftax_test.py"],
    imports = ["../.."],
    deps = [
        ":craftax",
        ":craftax_registration",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_library(
    name = "oracle",
    testonly = True,
    srcs = ["oracle.py"],
    data = [
        ":craftax_debug",
        ":oracle_textures",
    ],
    deps = [
        ":craftax",
        "@craftax_upstream//:oracle",
        craftax_oracle_requirement("flax"),
        craftax_oracle_requirement("imageio"),
        craftax_oracle_requirement("jax"),
        craftax_oracle_requirement("jaxlib"),
        craftax_oracle_requirement("matplotlib"),
        craftax_oracle_requirement("numpy"),
        craftax_oracle_requirement("pillow"),
    ],
)

py_binary(
    name = "generate_oracle_cache",
    testonly = True,
    srcs = ["generate_oracle_cache.py"],
    deps = [
        "@craftax_upstream//:oracle",
        craftax_oracle_requirement("jax"),
        craftax_oracle_requirement("flax"),
        craftax_oracle_requirement("imageio"),
        craftax_oracle_requirement("matplotlib"),
        craftax_oracle_requirement("pillow"),
    ],
)

genrule(
    name = "oracle_textures",
    testonly = True,
    outs = [
        "oracle_textures/classic.bz2",
        "oracle_textures/full.bz2",
    ],
    cmd = "$(location :generate_oracle_cache) --classic $(location oracle_textures/classic.bz2) --full $(location oracle_textures/full.bz2)",
    tools = [":generate_oracle_cache"],
)

py_test(
    name = "craftax_align_test",
    size = "enormous",
    timeout = "long",
    srcs = ["craftax_align_test.py"],
    imports = ["../.."],
    shard_count = 2,
    deps = [
        ":craftax_registration",
        ":oracle",
        requirement("absl-py"),
    ],
)

py_test(
    name = "craftax_behavior_test",
    size = "enormous",
    timeout = "long",
    srcs = ["craftax_behavior_test.py"],
    imports = ["../.."],
    shard_count = 2,
    deps = [
        ":oracle",
        requirement("absl-py"),
    ],
)
