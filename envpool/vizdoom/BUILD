# Copyright 2022 Garena Online Private Limited
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

genrule(
    name = "gen_pretrain_weight",
    srcs = [
        "@pretrain_weight//:pth",
    ],
    outs = [
        "policy-d1.pth",
        "policy-d3.pth",
    ],
    cmd = "cp $(SRCS) $(@D)",
)

genrule(
    name = "gen_vizdoom_maps",
    srcs = [
        "@vizdoom_lib//:vizdoom_maps",
        "@vizdoom_extra_maps//:vizdoom_maps",
    ],
    outs = [
        "maps/D1_basic.cfg",
        "maps/D1_basic.wad",
        "maps/D2_navigation.cfg",
        "maps/D2_navigation.wad",
        "maps/D3_battle.cfg",
        "maps/D3_battle.wad",
        "maps/D4_battle2.cfg",
        "maps/D4_battle2.wad",
        "maps/basic.cfg",
        "maps/basic.wad",
        "maps/basic_audio.cfg",
        "maps/basic_audio.wad",
        "maps/basic_notifications.cfg",
        "maps/basic_notifications.wad",
        "maps/cig.cfg",
        "maps/cig.wad",
        "maps/cig_with_unknown.wad",
        "maps/deadly_corridor.cfg",
        "maps/deadly_corridor.wad",
        "maps/deathmatch.cfg",
        "maps/deathmatch.wad",
        "maps/defend_the_center.cfg",
        "maps/defend_the_center.wad",
        "maps/defend_the_line.cfg",
        "maps/defend_the_line.wad",
        "maps/doom.cfg",
        "maps/doom2.cfg",
        "maps/freedoom1.cfg",
        "maps/freedoom2.cfg",
        "maps/health_gathering.cfg",
        "maps/health_gathering.wad",
        "maps/health_gathering_supreme.cfg",
        "maps/health_gathering_supreme.wad",
        "maps/learning.cfg",
        "maps/multi.cfg",
        "maps/multi_deathmatch.wad",
        "maps/multi_duel.cfg",
        "maps/multi_duel.wad",
        "maps/my_way_home.cfg",
        "maps/my_way_home.wad",
        "maps/oblige.cfg",
        "maps/predict_position.cfg",
        "maps/predict_position.wad",
        "maps/rocket_basic.cfg",
        "maps/rocket_basic.wad",
        "maps/simpler_basic.cfg",
        "maps/simpler_basic.wad",
        "maps/take_cover.cfg",
        "maps/take_cover.wad",
    ],
    cmd = """
mkdir -p "$(@D)/maps"
for src in $(SRCS); do
  cp "$${src}" "$(@D)/maps/$$(basename "$${src}")"
done
""",
)

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
    deps = ["@vizdoom_lib"],
)

cc_library(
    name = "vizdoom_env",
    hdrs = ["vizdoom_env.h"],
    deps = [
        ":utils",
        "//envpool/core:async_envpool",
        "//envpool/utils:image_process",
    ],
)

pybind_extension(
    name = "vizdoom_envpool",
    srcs = ["vizdoom_envpool.cc"],
    linkopts = [
        "-ldl",
    ],
    deps = [
        ":vizdoom_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "vizdoom",
    srcs = ["__init__.py"],
    data = [
        ":gen_vizdoom_maps",
        ":vizdoom_envpool.so",
        "//envpool/vizdoom/bin:freedoom",
        "//envpool/vizdoom/bin:vizdoom_bin",
        "//envpool/vizdoom/bin:vizdoom_pk3",
    ],
    deps = ["//envpool/python:api"],
)

py_library(
    name = "vizdoom_registration",
    srcs = ["registration.py"],
    deps = ["//envpool:registration"],
)

py_test(
    name = "vizdoom_test",
    size = "enormous",
    srcs = ["vizdoom_test.py"],
    deps = [
        ":vizdoom",
        ":vizdoom_registration",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("opencv-python-headless"),
    ],
)

py_test(
    name = "vizdoom_pretrain_test",
    size = "enormous",
    srcs = ["vizdoom_pretrain_test.py"],
    data = [":gen_pretrain_weight"],
    deps = [
        ":vizdoom",
        ":vizdoom_registration",
        "//envpool/atari:atari_network",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("tianshou"),
        requirement("torch"),
    ],
)
