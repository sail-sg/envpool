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

exports_files(["__init__.py"])

filegroup(
    name = "locomotion_metadata_source",
    srcs = [
        "locomotion/soccer/__init__.py",
        "locomotion/soccer/pitch.py",
        "locomotion/walkers/cmu_humanoid.py",
    ] + glob(["locomotion/examples/*.py"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "locomotion_assets",
    srcs = [
        "composer/arena.xml",
        "locomotion/walkers/assets/humanoid_CMU_V2019.xml",
        "locomotion/walkers/assets/humanoid_CMU_V2020.xml",
        "locomotion/walkers/assets/rodent.xml",
        "locomotion/walkers/assets/rodent_walker_skin.skn",
        "third_party/ant/LICENSE",
        "third_party/ant/ant.xml",
    ] + glob([
        "locomotion/arenas/assets/outdoor_natural/*",
        "locomotion/soccer/assets/boxhead/**",
        "locomotion/soccer/assets/humanoid/**",
        "locomotion/soccer/assets/pitch/**",
        "locomotion/soccer/assets/soccer_ball/**",
    ]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "locomotion_mocap_source",
    srcs = [
        "locomotion/tasks/reference_pose/cmu_subsets.py",
        "locomotion/walkers/initializers/mocap.py",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "mujoco_dmc_xml",
    srcs = glob(
        [
            "suite/*.xml",
            "suite/common/**",
            "suite/dog_assets/*",
        ],
        exclude = [
            "suite/common/__init__.py",
            "suite/dog_assets/BONELingual_bone_1.stl",
            "suite/dog_assets/BONELingual_bone_2.stl",
            "suite/dog_assets/BONELingual_bone_3.stl",
            "suite/dog_assets/BONELingual_bone_4.stl",
            "suite/dog_assets/BONELingual_bone_5.stl",
            "suite/dog_assets/BONELingual_bone_6.stl",
            "suite/dog_assets/BONELingual_bone_7.stl",
            "suite/dog_assets/BONELingual_bone_8.stl",
            "suite/dog_assets/BONELingual_bone_9.stl",
            "suite/dog_assets/BONEXiphoid_cartilage.stl",
            "suite/dog_assets/SKINbody.stl",
            "suite/dog_assets/dog_skin.msh",
        ],
    ),
    visibility = ["//visibility:public"],
)
