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
"""MiniGrid env registration."""

from __future__ import annotations

from collections.abc import Mapping

from envpool.registration import register

_IMPORT_PATH = "envpool.minigrid"
_SPEC_CLS = "MiniGridEnvSpec"
_DM_CLS = "MiniGridDMEnvPool"
_GYM_CLS = "MiniGridGymEnvPool"
_GYMNASIUM_CLS = "MiniGridGymnasiumEnvPool"
_BABYAI_MISSION_BYTES = 512
_RANDOM_START = {"agent_start_pos": (-1, -1), "agent_start_dir": -1}


def _register(task_id: str, max_episode_steps: int, **kwargs: object) -> None:
    register(
        task_id=task_id,
        import_path=_IMPORT_PATH,
        spec_cls=_SPEC_CLS,
        dm_cls=_DM_CLS,
        gym_cls=_GYM_CLS,
        gymnasium_cls=_GYMNASIUM_CLS,
        max_episode_steps=max_episode_steps,
        **kwargs,
    )


def _register_babyai(task_id: str, **kwargs: object) -> None:
    config: dict[str, object] = {
        "room_size": 8,
        "num_rows": 3,
        "num_cols": 3,
        "num_dists": 18,
        "num_objs": 8,
        "mission_bytes": _BABYAI_MISSION_BYTES,
    }
    config.update(kwargs)
    _register(task_id, 0, **config)


def _register_babyai_variants(
    env_name: str,
    variants: tuple[tuple[str, Mapping[str, object]], ...],
) -> None:
    for task_id, config in variants:
        _register_babyai(task_id, env_name=env_name, **config)


for size in (5, 6, 8, 16):
    _register(
        f"MiniGrid-Empty-{size}x{size}-v0",
        4 * size**2,
        env_name="empty",
        size=size,
    )
for size in (5, 6):
    _register(
        f"MiniGrid-Empty-Random-{size}x{size}-v0",
        4 * size**2,
        env_name="empty",
        size=size,
        **_RANDOM_START,
    )

for size in (5, 6, 8, 16):
    _register(
        f"MiniGrid-DoorKey-{size}x{size}-v0",
        10 * size**2,
        env_name="doorkey",
        size=size,
    )

for task_id, strip2_row in (
    ("MiniGrid-DistShift1-v0", 2),
    ("MiniGrid-DistShift2-v0", 5),
):
    _register(
        task_id,
        4 * 9 * 7,
        env_name="distshift",
        width=9,
        height=7,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        strip2_row=strip2_row,
    )

for task_id, size, num_crossings in (
    ("MiniGrid-LavaCrossingS9N1-v0", 9, 1),
    ("MiniGrid-LavaCrossingS9N2-v0", 9, 2),
    ("MiniGrid-LavaCrossingS9N3-v0", 9, 3),
    ("MiniGrid-LavaCrossingS11N5-v0", 11, 5),
):
    _register(
        task_id,
        4 * size**2,
        env_name="crossing",
        size=size,
        num_crossings=num_crossings,
        obstacle_type="lava",
    )

for task_id, size, num_crossings in (
    ("MiniGrid-SimpleCrossingS9N1-v0", 9, 1),
    ("MiniGrid-SimpleCrossingS9N2-v0", 9, 2),
    ("MiniGrid-SimpleCrossingS9N3-v0", 9, 3),
    ("MiniGrid-SimpleCrossingS11N5-v0", 11, 5),
):
    _register(
        task_id,
        4 * size**2,
        env_name="crossing",
        size=size,
        num_crossings=num_crossings,
        obstacle_type="wall",
    )

for size in (5, 6, 7):
    _register(
        f"MiniGrid-LavaGapS{size}-v0",
        4 * size**2,
        env_name="lava_gap",
        size=size,
        obstacle_type="lava",
    )

for task_id, size, n_obstacles, random_start in (
    ("MiniGrid-Dynamic-Obstacles-5x5-v0", 5, 2, False),
    ("MiniGrid-Dynamic-Obstacles-Random-5x5-v0", 5, 2, True),
    ("MiniGrid-Dynamic-Obstacles-6x6-v0", 6, 3, False),
    ("MiniGrid-Dynamic-Obstacles-Random-6x6-v0", 6, 3, True),
    ("MiniGrid-Dynamic-Obstacles-8x8-v0", 8, 4, False),
    ("MiniGrid-Dynamic-Obstacles-16x16-v0", 16, 8, False),
):
    config = {
        "env_name": "dynamic_obstacles",
        "size": size,
        "n_obstacles": n_obstacles,
        "action_max": 2,
    }
    if random_start:
        config.update(_RANDOM_START)
    _register(task_id, 4 * size**2, **config)

for task_id, size, num_objs in (
    ("MiniGrid-Fetch-5x5-N2-v0", 5, 2),
    ("MiniGrid-Fetch-6x6-N2-v0", 6, 2),
    ("MiniGrid-Fetch-8x8-N3-v0", 8, 3),
):
    _register(
        task_id,
        5 * size**2,
        env_name="fetch",
        size=size,
        num_objs=num_objs,
    )

_register("MiniGrid-FourRooms-v0", 100, env_name="four_rooms")

for size in (5, 6, 8):
    _register(
        f"MiniGrid-GoToDoor-{size}x{size}-v0",
        4 * size**2,
        env_name="goto_door",
        size=size,
    )

for task_id, size, num_objs in (
    ("MiniGrid-GoToObject-6x6-N2-v0", 6, 2),
    ("MiniGrid-GoToObject-8x8-N2-v0", 8, 2),
):
    _register(
        task_id,
        5 * size**2,
        env_name="goto_object",
        size=size,
        num_objs=num_objs,
    )

for task_id, room_size, num_rows in (
    ("MiniGrid-KeyCorridorS3R1-v0", 3, 1),
    ("MiniGrid-KeyCorridorS3R2-v0", 3, 2),
    ("MiniGrid-KeyCorridorS3R3-v0", 3, 3),
    ("MiniGrid-KeyCorridorS4R3-v0", 4, 3),
    ("MiniGrid-KeyCorridorS5R3-v0", 5, 3),
    ("MiniGrid-KeyCorridorS6R3-v0", 6, 3),
):
    _register(
        task_id,
        30 * room_size**2,
        env_name="key_corridor",
        room_size=room_size,
        num_rows=num_rows,
        obj_type="ball",
    )

_register(
    "MiniGrid-LockedRoom-v0",
    10 * 19,
    env_name="locked_room",
    size=19,
)

for task_id, size, random_length in (
    ("MiniGrid-MemoryS17Random-v0", 17, True),
    ("MiniGrid-MemoryS13Random-v0", 13, True),
    ("MiniGrid-MemoryS13-v0", 13, False),
    ("MiniGrid-MemoryS11-v0", 11, False),
    ("MiniGrid-MemoryS9-v0", 9, False),
    ("MiniGrid-MemoryS7-v0", 7, False),
):
    _register(
        task_id,
        5 * size**2,
        env_name="memory",
        size=size,
        random_length=random_length,
    )

for task_id, min_num_rooms, max_num_rooms, max_room_size in (
    ("MiniGrid-MultiRoom-N2-S4-v0", 2, 2, 4),
    ("MiniGrid-MultiRoom-N4-S5-v0", 6, 6, 5),
    ("MiniGrid-MultiRoom-N6-v0", 6, 6, 10),
):
    _register(
        task_id,
        max_num_rooms * 20,
        env_name="multi_room",
        min_num_rooms=min_num_rooms,
        max_num_rooms=max_num_rooms,
        max_room_size=max_room_size,
    )

for (
    task_id,
    env_name,
    key_in_box,
    blocked,
    agent_room,
    num_quarters,
    num_rooms_visited,
) in (
    (
        "MiniGrid-ObstructedMaze-1Dl-v0",
        "obstructed_maze_1dlhb",
        False,
        False,
        (0, 0),
        1,
        2,
    ),
    (
        "MiniGrid-ObstructedMaze-1Dlh-v0",
        "obstructed_maze_1dlhb",
        True,
        False,
        (0, 0),
        1,
        2,
    ),
    (
        "MiniGrid-ObstructedMaze-1Dlhb-v0",
        "obstructed_maze_1dlhb",
        True,
        True,
        (0, 0),
        1,
        2,
    ),
    (
        "MiniGrid-ObstructedMaze-2Dl-v0",
        "obstructed_maze_full",
        False,
        False,
        (2, 1),
        1,
        4,
    ),
    (
        "MiniGrid-ObstructedMaze-2Dlh-v0",
        "obstructed_maze_full",
        True,
        False,
        (2, 1),
        1,
        4,
    ),
    (
        "MiniGrid-ObstructedMaze-2Dlhb-v0",
        "obstructed_maze_full",
        True,
        True,
        (2, 1),
        1,
        4,
    ),
    (
        "MiniGrid-ObstructedMaze-1Q-v0",
        "obstructed_maze_full",
        True,
        True,
        (1, 1),
        1,
        5,
    ),
    (
        "MiniGrid-ObstructedMaze-2Q-v0",
        "obstructed_maze_full",
        True,
        True,
        (2, 1),
        2,
        11,
    ),
    (
        "MiniGrid-ObstructedMaze-Full-v0",
        "obstructed_maze_full",
        True,
        True,
        (2, 1),
        4,
        25,
    ),
    (
        "MiniGrid-ObstructedMaze-2Dlhb-v1",
        "obstructed_maze_full_v1",
        True,
        True,
        (2, 1),
        1,
        4,
    ),
    (
        "MiniGrid-ObstructedMaze-1Q-v1",
        "obstructed_maze_full_v1",
        True,
        True,
        (1, 1),
        1,
        5,
    ),
    (
        "MiniGrid-ObstructedMaze-2Q-v1",
        "obstructed_maze_full_v1",
        True,
        True,
        (2, 1),
        2,
        11,
    ),
    (
        "MiniGrid-ObstructedMaze-Full-v1",
        "obstructed_maze_full_v1",
        True,
        True,
        (2, 1),
        4,
        25,
    ),
):
    _register(
        task_id,
        4 * num_rooms_visited * 6**2,
        env_name=env_name,
        agent_room=agent_room,
        key_in_box=key_in_box,
        blocked=blocked,
        num_quarters=num_quarters,
        num_rooms_visited=num_rooms_visited,
    )

_register("MiniGrid-Playground-v0", 100, env_name="playground")

for task_id, size, num_objs in (
    ("MiniGrid-PutNear-6x6-N2-v0", 6, 2),
    ("MiniGrid-PutNear-8x8-N3-v0", 8, 3),
):
    _register(
        task_id,
        5 * size,
        env_name="put_near",
        size=size,
        num_objs=num_objs,
    )

for size in (6, 8):
    _register(
        f"MiniGrid-RedBlueDoors-{size}x{size}-v0",
        20 * size**2,
        env_name="red_blue_door",
        size=size,
    )

_register("MiniGrid-Unlock-v0", 8 * 6**2, env_name="unlock")
_register("MiniGrid-UnlockPickup-v0", 8 * 6**2, env_name="unlock_pickup")
_register(
    "MiniGrid-BlockedUnlockPickup-v0",
    16 * 6**2,
    env_name="blocked_unlock_pickup",
)

_register_babyai("BabyAI-ActionObjDoor-v0", env_name="babyai_action_obj_door")
_register_babyai(
    "BabyAI-BlockedUnlockPickup-v0",
    env_name="babyai_blocked_unlock_pickup",
)
_register_babyai("BabyAI-BossLevel-v0", env_name="babyai_boss_level")
_register_babyai(
    "BabyAI-BossLevelNoUnlock-v0",
    env_name="babyai_boss_level_no_unlock",
)

for task_id, room_size in (
    ("BabyAI-FindObjS5-v0", 5),
    ("BabyAI-FindObjS6-v0", 6),
    ("BabyAI-FindObjS7-v0", 7),
):
    _register_babyai(
        task_id,
        env_name="babyai_find_obj",
        room_size=room_size,
    )

_register_babyai_variants(
    "babyai_goto",
    (
        ("BabyAI-GoTo-v0", {}),
        ("BabyAI-GoToOpen-v0", {"doors_open": True}),
        ("BabyAI-GoToObjMaze-v0", {"doors_open": False, "num_dists": 1}),
        (
            "BabyAI-GoToObjMazeOpen-v0",
            {"doors_open": True, "num_dists": 1},
        ),
        ("BabyAI-GoToObjMazeS4-v0", {"num_dists": 1, "room_size": 4}),
        (
            "BabyAI-GoToObjMazeS4R2-v0",
            {"num_cols": 2, "num_dists": 1, "num_rows": 2, "room_size": 4},
        ),
        ("BabyAI-GoToObjMazeS5-v0", {"num_dists": 1, "room_size": 5}),
        ("BabyAI-GoToObjMazeS6-v0", {"num_dists": 1, "room_size": 6}),
        ("BabyAI-GoToObjMazeS7-v0", {"num_dists": 1, "room_size": 7}),
    ),
)

_register_babyai("BabyAI-GoToDoor-v0", env_name="babyai_goto_door")
_register_babyai(
    "BabyAI-GoToImpUnlock-v0",
    env_name="babyai_goto_imp_unlock",
)

_register_babyai_variants(
    "babyai_goto_local",
    (
        ("BabyAI-GoToLocal-v0", {"num_dists": 8}),
        ("BabyAI-GoToLocalS5N2-v0", {"num_dists": 2, "room_size": 5}),
        ("BabyAI-GoToLocalS6N2-v0", {"num_dists": 2, "room_size": 6}),
        ("BabyAI-GoToLocalS6N3-v0", {"num_dists": 3, "room_size": 6}),
        ("BabyAI-GoToLocalS6N4-v0", {"num_dists": 4, "room_size": 6}),
        ("BabyAI-GoToLocalS7N4-v0", {"num_dists": 4, "room_size": 7}),
        ("BabyAI-GoToLocalS7N5-v0", {"num_dists": 5, "room_size": 7}),
        ("BabyAI-GoToLocalS8N2-v0", {"num_dists": 2, "room_size": 8}),
        ("BabyAI-GoToLocalS8N3-v0", {"num_dists": 3, "room_size": 8}),
        ("BabyAI-GoToLocalS8N4-v0", {"num_dists": 4, "room_size": 8}),
        ("BabyAI-GoToLocalS8N5-v0", {"num_dists": 5, "room_size": 8}),
        ("BabyAI-GoToLocalS8N6-v0", {"num_dists": 6, "room_size": 8}),
        ("BabyAI-GoToLocalS8N7-v0", {"num_dists": 7, "room_size": 8}),
    ),
)

_register_babyai_variants(
    "babyai_goto_obj",
    (
        ("BabyAI-GoToObj-v0", {}),
        ("BabyAI-GoToObjS4-v0", {"room_size": 4}),
        ("BabyAI-GoToObjS6-v1", {"room_size": 6}),
    ),
)

_register_babyai(
    "BabyAI-GoToObjDoor-v0",
    env_name="babyai_goto_obj_door",
)
_register_babyai(
    "BabyAI-GoToRedBall-v0",
    env_name="babyai_goto_red_ball",
    num_dists=7,
)
_register_babyai(
    "BabyAI-GoToRedBallGrey-v0",
    env_name="babyai_goto_red_ball_grey",
    num_dists=7,
)
_register_babyai(
    "BabyAI-GoToRedBallNoDists-v0",
    env_name="babyai_goto_red_ball_no_dists",
    num_dists=0,
)
_register_babyai(
    "BabyAI-GoToRedBlueBall-v0",
    env_name="babyai_goto_red_blue_ball",
    num_dists=7,
)

_register_babyai_variants(
    "babyai_goto_seq",
    (
        ("BabyAI-GoToSeq-v0", {}),
        (
            "BabyAI-GoToSeqS5R2-v0",
            {"num_cols": 2, "num_dists": 4, "num_rows": 2, "room_size": 5},
        ),
    ),
)

_register_babyai_variants(
    "babyai_key_corridor",
    (
        ("BabyAI-KeyCorridor-v0", {"room_size": 6}),
        ("BabyAI-KeyCorridorS3R1-v0", {"num_rows": 1, "room_size": 3}),
        ("BabyAI-KeyCorridorS3R2-v0", {"num_rows": 2, "room_size": 3}),
        ("BabyAI-KeyCorridorS3R3-v0", {"num_rows": 3, "room_size": 3}),
        ("BabyAI-KeyCorridorS4R3-v0", {"num_rows": 3, "room_size": 4}),
        ("BabyAI-KeyCorridorS5R3-v0", {"num_rows": 3, "room_size": 5}),
        ("BabyAI-KeyCorridorS6R3-v0", {"num_rows": 3, "room_size": 6}),
    ),
)

_register_babyai("BabyAI-KeyInBox-v0", env_name="babyai_key_in_box")
_register_babyai(
    "BabyAI-MiniBossLevel-v0",
    env_name="babyai_mini_boss_level",
)

_register_babyai_variants(
    "babyai_move_two_across",
    (
        ("BabyAI-MoveTwoAcrossS5N2-v0", {"objs_per_room": 2, "room_size": 5}),
        ("BabyAI-MoveTwoAcrossS8N9-v0", {"objs_per_room": 9, "room_size": 8}),
    ),
)

_register_babyai_variants(
    "babyai_one_room",
    (
        ("BabyAI-OneRoomS8-v0", {}),
        ("BabyAI-OneRoomS12-v0", {"room_size": 12}),
        ("BabyAI-OneRoomS16-v0", {"room_size": 16}),
        ("BabyAI-OneRoomS20-v0", {"room_size": 20}),
    ),
)

_register_babyai("BabyAI-Open-v0", env_name="babyai_open")

_register_babyai_variants(
    "babyai_open_door",
    (
        ("BabyAI-OpenDoor-v0", {}),
        ("BabyAI-OpenDoorColor-v0", {"select_by": "color"}),
        ("BabyAI-OpenDoorDebug-v0", {"debug": True, "select_by": ""}),
        ("BabyAI-OpenDoorLoc-v0", {"select_by": "loc"}),
    ),
)

_register_babyai_variants(
    "babyai_open_doors_order",
    (
        ("BabyAI-OpenDoorsOrderN2-v0", {"num_doors": 2}),
        ("BabyAI-OpenDoorsOrderN2Debug-v0", {"debug": True, "num_doors": 2}),
        ("BabyAI-OpenDoorsOrderN4-v0", {"num_doors": 4}),
        ("BabyAI-OpenDoorsOrderN4Debug-v0", {"debug": True, "num_doors": 4}),
    ),
)

_register_babyai(
    "BabyAI-OpenRedBlueDoors-v0",
    env_name="babyai_open_two_doors",
    first_color="red",
    second_color="blue",
)
_register_babyai(
    "BabyAI-OpenRedBlueDoorsDebug-v0",
    env_name="babyai_open_two_doors",
    first_color="red",
    second_color="blue",
    strict=True,
)
_register_babyai(
    "BabyAI-OpenRedDoor-v0",
    env_name="babyai_open_red_door",
)
_register_babyai(
    "BabyAI-OpenTwoDoors-v0",
    env_name="babyai_open_two_doors",
)

_register_babyai("BabyAI-Pickup-v0", env_name="babyai_pickup")
_register_babyai(
    "BabyAI-PickupAbove-v0",
    env_name="babyai_pickup_above",
)
_register_babyai("BabyAI-PickupDist-v0", env_name="babyai_pickup_dist")
_register_babyai(
    "BabyAI-PickupDistDebug-v0",
    env_name="babyai_pickup_dist",
    debug=True,
)
_register_babyai("BabyAI-PickupLoc-v0", env_name="babyai_pickup_loc")

_register_babyai_variants(
    "babyai_put_next_local",
    (
        ("BabyAI-PutNextLocal-v0", {}),
        ("BabyAI-PutNextLocalS5N3-v0", {"num_objs": 3, "room_size": 5}),
        ("BabyAI-PutNextLocalS6N4-v0", {"num_objs": 4, "room_size": 6}),
    ),
)

_register_babyai_variants(
    "babyai_put_next",
    (
        ("BabyAI-PutNextS4N1-v0", {"objs_per_room": 1, "room_size": 4}),
        ("BabyAI-PutNextS5N1-v0", {"objs_per_room": 1, "room_size": 5}),
        ("BabyAI-PutNextS5N2-v0", {"objs_per_room": 2, "room_size": 5}),
        (
            "BabyAI-PutNextS5N2Carrying-v0",
            {"objs_per_room": 2, "room_size": 5, "start_carrying": True},
        ),
        ("BabyAI-PutNextS6N3-v0", {"objs_per_room": 3, "room_size": 6}),
        (
            "BabyAI-PutNextS6N3Carrying-v0",
            {"objs_per_room": 3, "room_size": 6, "start_carrying": True},
        ),
        ("BabyAI-PutNextS7N4-v0", {"objs_per_room": 4, "room_size": 7}),
        (
            "BabyAI-PutNextS7N4Carrying-v0",
            {"objs_per_room": 4, "room_size": 7, "start_carrying": True},
        ),
    ),
)

_register_babyai("BabyAI-Synth-v0", env_name="babyai_synth")
_register_babyai("BabyAI-SynthLoc-v0", env_name="babyai_synth_loc")
_register_babyai(
    "BabyAI-SynthS5R2-v0",
    env_name="babyai_synth",
    num_rows=2,
    room_size=5,
)
_register_babyai("BabyAI-SynthSeq-v0", env_name="babyai_synth_seq")
_register_babyai(
    "BabyAI-UnblockPickup-v0",
    env_name="babyai_unblock_pickup",
)
_register_babyai("BabyAI-Unlock-v0", env_name="babyai_unlock")
_register_babyai(
    "BabyAI-UnlockLocal-v0",
    env_name="babyai_unlock_local",
)
_register_babyai(
    "BabyAI-UnlockLocalDist-v0",
    env_name="babyai_unlock_local",
    distractors=True,
)
_register_babyai(
    "BabyAI-UnlockPickup-v0",
    env_name="babyai_unlock_pickup",
)
_register_babyai(
    "BabyAI-UnlockPickupDist-v0",
    env_name="babyai_unlock_pickup",
    distractors=True,
)
_register_babyai(
    "BabyAI-UnlockToUnlock-v0",
    env_name="babyai_unlock_to_unlock",
)
