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

load("@rules_cc//cc:defs.bzl", "cc_library")

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

cc_library(
    name = "box2d",
    srcs = glob(["include/box2d/*.h"]) + [
        "src/collision/b2_broad_phase.cpp",
        "src/collision/b2_chain_shape.cpp",
        "src/collision/b2_circle_shape.cpp",
        "src/collision/b2_collide_circle.cpp",
        "src/collision/b2_collide_edge.cpp",
        "src/collision/b2_collide_polygon.cpp",
        "src/collision/b2_collision.cpp",
        "src/collision/b2_distance.cpp",
        "src/collision/b2_dynamic_tree.cpp",
        "src/collision/b2_edge_shape.cpp",
        "src/collision/b2_polygon_shape.cpp",
        "src/collision/b2_time_of_impact.cpp",
        "src/common/b2_block_allocator.cpp",
        "src/common/b2_draw.cpp",
        "src/common/b2_math.cpp",
        "src/common/b2_settings.cpp",
        "src/common/b2_stack_allocator.cpp",
        "src/common/b2_timer.cpp",
        "src/dynamics/b2_body.cpp",
        "src/dynamics/b2_chain_circle_contact.cpp",
        "src/dynamics/b2_chain_circle_contact.h",
        "src/dynamics/b2_chain_polygon_contact.cpp",
        "src/dynamics/b2_chain_polygon_contact.h",
        "src/dynamics/b2_circle_contact.cpp",
        "src/dynamics/b2_circle_contact.h",
        "src/dynamics/b2_contact.cpp",
        "src/dynamics/b2_contact_manager.cpp",
        "src/dynamics/b2_contact_solver.cpp",
        "src/dynamics/b2_contact_solver.h",
        "src/dynamics/b2_distance_joint.cpp",
        "src/dynamics/b2_edge_circle_contact.cpp",
        "src/dynamics/b2_edge_circle_contact.h",
        "src/dynamics/b2_edge_polygon_contact.cpp",
        "src/dynamics/b2_edge_polygon_contact.h",
        "src/dynamics/b2_fixture.cpp",
        "src/dynamics/b2_friction_joint.cpp",
        "src/dynamics/b2_gear_joint.cpp",
        "src/dynamics/b2_island.cpp",
        "src/dynamics/b2_island.h",
        "src/dynamics/b2_joint.cpp",
        "src/dynamics/b2_motor_joint.cpp",
        "src/dynamics/b2_mouse_joint.cpp",
        "src/dynamics/b2_polygon_circle_contact.cpp",
        "src/dynamics/b2_polygon_circle_contact.h",
        "src/dynamics/b2_polygon_contact.cpp",
        "src/dynamics/b2_polygon_contact.h",
        "src/dynamics/b2_prismatic_joint.cpp",
        "src/dynamics/b2_pulley_joint.cpp",
        "src/dynamics/b2_revolute_joint.cpp",
        "src/dynamics/b2_weld_joint.cpp",
        "src/dynamics/b2_wheel_joint.cpp",
        "src/dynamics/b2_world.cpp",
        "src/dynamics/b2_world_callbacks.cpp",
        "src/rope/b2_rope.cpp",
    ],
    hdrs = glob(["include/box2d/*.h"]),
    includes = [
        "include",
        "src",
    ],
    linkopts = select({
        ":windows": [],
        "//conditions:default": [
            "-ldl",
        ],
    }),
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
