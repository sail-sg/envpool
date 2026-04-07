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

filegroup(
    name = "mujoco_dmc_xml",
    srcs = glob([
        "suite/*.xml",
        "suite/common/**",
        "suite/dog_assets/*",
    ], exclude = [
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
    ]),
    visibility = ["//visibility:public"],
)
