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

UNUSED_GYMNASIUM_ROBOTICS_ASSETS = [
    "assets/LICENSE.md",
    "assets/adroit_hand/resources/meshes/arm_base.stl",
    "assets/adroit_hand/resources/meshes/arm_trunk.stl",
    "assets/adroit_hand/resources/meshes/arm_trunk_asmbly.stl",
    "assets/adroit_hand/resources/meshes/distal_ellipsoid.stl",
    "assets/adroit_hand/resources/meshes/elbow_flex.stl",
    "assets/adroit_hand/resources/meshes/elbow_rotate_motor.stl",
    "assets/adroit_hand/resources/meshes/elbow_rotate_muscle.stl",
    "assets/adroit_hand/resources/meshes/forearm_Cy_PlateAsmbly(muscle_cone).stl",
    "assets/adroit_hand/resources/meshes/forearm_Cy_PlateAsmbly.stl",
    "assets/adroit_hand/resources/meshes/forearm_PlateAsmbly.stl",
    "assets/adroit_hand/resources/meshes/forearm_electric.stl",
    "assets/adroit_hand/resources/meshes/forearm_electric_cvx.stl",
    "assets/adroit_hand/resources/meshes/forearm_muscle.stl",
    "assets/adroit_hand/resources/meshes/forearm_simple_cvx.stl",
    "assets/adroit_hand/resources/meshes/forearm_weight.stl",
    "assets/adroit_hand/resources/meshes/upper_arm.stl",
    "assets/adroit_hand/resources/meshes/upper_arm_asmbl_shoulder.stl",
    "assets/adroit_hand/resources/meshes/upper_arm_ass.stl",
    "assets/adroit_hand/resources/textures/darkwood.png",
    "assets/adroit_hand/resources/textures/dice.png",
    "assets/adroit_hand/resources/textures/foil.png",
    "assets/adroit_hand/resources/textures/skin.png",
    "assets/kitchen_franka/franka_assets/LICENSE.md",
    "assets/kitchen_franka/franka_assets/basic_scene.xml",
    "assets/kitchen_franka/franka_assets/franka_config.xml",
    "assets/kitchen_franka/franka_assets/meshes/visual/finger.stl",
    "assets/kitchen_franka/kitchen_assets/LICENSE.md",
    "assets/kitchen_franka/kitchen_assets/meshes/burnerplate_mesh.stl",
    "assets/kitchen_franka/kitchen_assets/meshes/handle2.stl",
    "assets/kitchen_franka/kitchen_assets/meshes/hingecabinet.stl",
    "assets/kitchen_franka/kitchen_assets/meshes/hingedoor.stl",
    "assets/kitchen_franka/kitchen_assets/meshes/hingehandle.stl",
    "assets/kitchen_franka/kitchen_assets/meshes/microefeet.stl",
    "assets/kitchen_franka/kitchen_assets/meshes/slidecabinet.stl",
    "assets/kitchen_franka/kitchen_assets/meshes/slidedoor.stl",
    "assets/kitchen_franka/kitchen_assets/meshes/tile.stl",
    "assets/kitchen_franka/kitchen_assets/meshes/wall.stl",
    "assets/kitchen_franka/kitchen_assets/textures/tile1.png",
    "assets/kitchen_franka/kitchen_assets/textures/white_marble_tile.png",
    "assets/point/point.xml",
]

filegroup(
    name = "gymnasium_robotics_assets",
    srcs = glob(
        ["assets/**"],
        exclude = UNUSED_GYMNASIUM_ROBOTICS_ASSETS,
    ),
    visibility = ["//visibility:public"],
)
