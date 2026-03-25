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

from envpool.registration import register

_IMPORT_PATH = "envpool.minigrid"
_SPEC_CLS = "MiniGridEnvSpec"
_DM_CLS = "MiniGridDMEnvPool"
_GYM_CLS = "MiniGridGymEnvPool"
_GYMNASIUM_CLS = "MiniGridGymnasiumEnvPool"
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
