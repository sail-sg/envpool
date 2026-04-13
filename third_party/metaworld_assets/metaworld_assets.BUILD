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

load("@rules_python//python:defs.bzl", "py_library")

_METAWORLD_TEXTURES = [
    "brick1.png",
    "floor2.png",
    "metal.png",
    "metal1.png",
    "metal2.png",
    "plaster1.png",
    "wood1.png",
    "wood2.png",
    "wood4.png",
]

_UNUSED_METAWORLD_ASSETS = [
    "metaworld/assets/objects/assets/block_cyl.xml",
    "metaworld/assets/objects/assets/club.xml",
    "metaworld/assets/objects/assets/club_dependencies.xml",
    "metaworld/assets/objects/assets/laptop.xml",
    "metaworld/assets/objects/assets/laptop_dependencies.xml",
    "metaworld/assets/objects/assets/peg_insert.xml",
    "metaworld/assets/objects/assets/plug_wall_dependencies.xml",
    "metaworld/assets/objects/assets/shelfb.xml",
    "metaworld/assets/objects/assets/shelfb_dependencies.xml",
    "metaworld/assets/objects/assets/table.xml",
    "metaworld/assets/objects/assets/table_dependencies.xml",
    "metaworld/assets/objects/assets/table_hole.xml",
    "metaworld/assets/objects/assets/window.xml",
    "metaworld/assets/objects/meshes/buttonbox/button.stl",
    "metaworld/assets/objects/meshes/buttonbox/buttonring.stl",
    "metaworld/assets/objects/meshes/coffeemachine/bodypiece2.stl",
    "metaworld/assets/objects/meshes/coffeemachine/bodypiece3.stl",
    "metaworld/assets/objects/meshes/coffeemachine/cup.stl",
    "metaworld/assets/objects/meshes/doorlock/handle.stl",
    "metaworld/assets/objects/meshes/golf_club/club_handle.stl",
    "metaworld/assets/objects/meshes/golf_club/club_head.stl",
    "metaworld/assets/objects/meshes/golf_club/club_tape.stl",
    "metaworld/assets/objects/meshes/hammer/hammerhead.stl",
    "metaworld/assets/objects/meshes/laptop/laptop_base.stl",
    "metaworld/assets/objects/meshes/laptop/laptop_hinge.stl",
    "metaworld/assets/objects/meshes/laptop/laptop_keys.stl",
    "metaworld/assets/objects/meshes/laptop/laptop_screen.stl",
    "metaworld/assets/objects/meshes/laptop/laptop_top.stl",
    "metaworld/assets/objects/meshes/shelfb/shelf_0.stl",
    "metaworld/assets/objects/meshes/shelfb/shelf_1.stl",
    "metaworld/assets/objects/meshes/shelfb/shelf_frame.stl",
    "metaworld/assets/objects/meshes/table/table_hole2.stl",
    "metaworld/assets/sawyer_xyz/sawyer_laptop.xml",
    "metaworld/assets/sawyer_xyz/sawyer_pick_and_place.xml",
    "metaworld/assets/sawyer_xyz/sawyer_reach_push_pick_and_place.xml",
    "metaworld/assets/sawyer_xyz/sawyer_reach_push_pick_and_place_wall.xml",
    "metaworld/assets/sawyer_xyz/sawyer_shelf_removing.xml",
    "metaworld/assets/sawyer_xyz/sawyer_sweep_tool.xml",
    "metaworld/assets/sawyer_xyz/sawyer_table_with_hole_no_puck.xml",
    "metaworld/assets/sawyer_xyz/sawyer_window.xml",
]

filegroup(
    name = "metaworld_assets",
    srcs = glob(
        [
            "metaworld/assets/objects/assets/*.xml",
            "metaworld/assets/objects/meshes/**",
            "metaworld/assets/sawyer_xyz/*.xml",
        ],
        exclude = _UNUSED_METAWORLD_ASSETS,
    ) + [
        "metaworld/assets/scene/basic_scene.xml",
        "metaworld/assets/scene/basic_scene_b.xml",
    ] + [
        "metaworld/assets/textures/%s" % texture
        for texture in _METAWORLD_TEXTURES
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "metaworld_oracle",
    srcs = glob(
        ["metaworld/**/*.py"],
        exclude = ["metaworld/assets/**"],
    ),
    data = [":metaworld_assets"],
    imports = ["."],
    visibility = ["//visibility:public"],
)
