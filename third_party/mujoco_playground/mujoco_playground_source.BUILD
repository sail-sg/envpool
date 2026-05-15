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

load("@envpool//envpool:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library")

filegroup(
    name = "runtime_assets",
    srcs = glob([
        "mujoco_playground/_src/locomotion/apollo/xmls/**",
        "mujoco_playground/_src/locomotion/berkeley_humanoid/xmls/**",
        "mujoco_playground/_src/locomotion/g1/xmls/**",
        "mujoco_playground/_src/locomotion/go1/xmls/**",
        "mujoco_playground/_src/locomotion/h1/xmls/**",
        "mujoco_playground/_src/locomotion/op3/xmls/**",
        "mujoco_playground/_src/locomotion/spot/xmls/**",
        "mujoco_playground/_src/locomotion/t1/xmls/**",
        "mujoco_playground/_src/manipulation/aloha/xmls/**",
        "mujoco_playground/_src/manipulation/aero_hand/xmls/**",
        "mujoco_playground/_src/manipulation/franka_emika_panda/xmls/**",
        "mujoco_playground/_src/manipulation/franka_emika_panda_robotiq/assets/**",
        "mujoco_playground/_src/manipulation/franka_emika_panda_robotiq/xmls/**",
        "mujoco_playground/_src/manipulation/leap_hand/xmls/**",
    ]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "python_source",
    srcs = glob(
        [
            "mujoco_playground/**/*.py",
        ],
        exclude = [
            "mujoco_playground/experimental/**",
        ],
    ),
    visibility = ["//visibility:public"],
)

py_library(
    name = "python",
    srcs = [":python_source"],
    data = [
        ":runtime_assets",
        "@mujoco_menagerie_playground//:aloha_assets",
        "@mujoco_menagerie_playground//:apptronik_apollo_assets",
        "@mujoco_menagerie_playground//:berkeley_humanoid_assets",
        "@mujoco_menagerie_playground//:boston_dynamics_spot_assets",
        "@mujoco_menagerie_playground//:booster_t1_assets",
        "@mujoco_menagerie_playground//:franka_emika_panda_assets",
        "@mujoco_menagerie_playground//:google_barkour_vb_assets",
        "@mujoco_menagerie_playground//:leap_hand_assets",
        "@mujoco_menagerie_playground//:robotis_op3_assets",
        "@mujoco_menagerie_playground//:robotiq_2f85_v4_assets",
        "@mujoco_menagerie_playground//:tetheria_aero_hand_open_assets",
        "@mujoco_menagerie_playground//:unitree_g1_assets",
        "@mujoco_menagerie_playground//:unitree_go1_assets",
        "@mujoco_menagerie_playground//:unitree_h1_assets",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "@envpool//third_party/mujoco_playground/stubs:mujoco_playground_oracle_stubs",
        requirement("etils"),
        requirement("jax"),
        requirement("mujoco"),
        requirement("mujoco-mjx"),
        requirement("numpy"),
        requirement("tqdm"),
    ],
)
