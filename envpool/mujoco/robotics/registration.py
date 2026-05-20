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
"""Gymnasium-Robotics env registration."""

from typing import Any

from envpool.registration import register

_IMPORT_PATH = "envpool.mujoco.robotics"
_ROBOTICS_ENV_CLASSES = {
    "Fetch": (
        "GymnasiumRoboticsFetchEnvSpec",
        "GymnasiumRoboticsFetchDMEnvPool",
        "GymnasiumRoboticsFetchGymnasiumEnvPool",
    ),
    "Hand": (
        "GymnasiumRoboticsHandEnvSpec",
        "GymnasiumRoboticsHandDMEnvPool",
        "GymnasiumRoboticsHandGymnasiumEnvPool",
    ),
    "Adroit": (
        "GymnasiumRoboticsAdroitEnvSpec",
        "GymnasiumRoboticsAdroitDMEnvPool",
        "GymnasiumRoboticsAdroitGymnasiumEnvPool",
    ),
    "PointMaze": (
        "GymnasiumRoboticsPointMazeEnvSpec",
        "GymnasiumRoboticsPointMazeDMEnvPool",
        "GymnasiumRoboticsPointMazeGymnasiumEnvPool",
    ),
    "Kitchen": (
        "GymnasiumRoboticsKitchenEnvSpec",
        "GymnasiumRoboticsKitchenDMEnvPool",
        "GymnasiumRoboticsKitchenGymnasiumEnvPool",
    ),
}

_FETCH_TASKS: dict[str, dict[str, Any]] = {
    "FetchReach": {
        "xml_file": "fetch/reach.xml",
        "has_object": False,
        "block_gripper": True,
        "target_in_the_air": True,
        "gripper_extra_height": 0.2,
        "target_offset_x": 0.0,
        "target_offset_y": 0.0,
        "target_offset_z": 0.0,
        "obj_range": 0.15,
        "target_range": 0.15,
        "distance_threshold": 0.05,
        "initial_slide0": 0.4049,
        "initial_slide1": 0.48,
        "initial_slide2": 0.0,
        "initial_object_x": 1.25,
        "initial_object_y": 0.53,
        "initial_object_z": 0.4,
    },
    "FetchPush": {
        "xml_file": "fetch/push.xml",
        "has_object": True,
        "block_gripper": True,
        "target_in_the_air": False,
        "gripper_extra_height": 0.0,
        "target_offset_x": 0.0,
        "target_offset_y": 0.0,
        "target_offset_z": 0.0,
        "obj_range": 0.15,
        "target_range": 0.15,
        "distance_threshold": 0.05,
        "initial_slide0": 0.4049,
        "initial_slide1": 0.48,
        "initial_slide2": 0.0,
        "initial_object_x": 1.25,
        "initial_object_y": 0.53,
        "initial_object_z": 0.4,
    },
    "FetchPickAndPlace": {
        "xml_file": "fetch/pick_and_place.xml",
        "has_object": True,
        "block_gripper": False,
        "target_in_the_air": True,
        "gripper_extra_height": 0.2,
        "target_offset_x": 0.0,
        "target_offset_y": 0.0,
        "target_offset_z": 0.0,
        "obj_range": 0.15,
        "target_range": 0.15,
        "distance_threshold": 0.05,
        "initial_slide0": 0.4049,
        "initial_slide1": 0.48,
        "initial_slide2": 0.0,
        "initial_object_x": 1.25,
        "initial_object_y": 0.53,
        "initial_object_z": 0.4,
    },
    "FetchSlide": {
        "xml_file": "fetch/slide.xml",
        "has_object": True,
        "block_gripper": True,
        "target_in_the_air": False,
        "gripper_extra_height": -0.02,
        "target_offset_x": 0.4,
        "target_offset_y": 0.0,
        "target_offset_z": 0.0,
        "obj_range": 0.1,
        "target_range": 0.3,
        "distance_threshold": 0.05,
        "initial_slide0": 0.4049,
        "initial_slide1": 0.48,
        "initial_slide2": 0.0,
        "initial_object_x": 1.7,
        "initial_object_y": 1.1,
        "initial_object_z": 0.41,
    },
}

_HAND_MANIPULATE_OBJECTS: dict[str, dict[str, Any]] = {
    "Block": {
        "xml_file": "hand/manipulate_block.xml",
        "touch_xml_file": "hand/manipulate_block_touch_sensors.xml",
        "distance_threshold": 0.01,
        "randomize_initial_rotation": True,
        "ignore_z_target_rotation": False,
        "variants": {
            "": {
                "target_position": "random",
                "target_rotation": "xyz",
            },
            "Full": {
                "target_position": "random",
                "target_rotation": "xyz",
            },
            "RotateParallel": {
                "target_position": "ignore",
                "target_rotation": "parallel",
            },
            "RotateXYZ": {
                "target_position": "ignore",
                "target_rotation": "xyz",
            },
            "RotateZ": {
                "target_position": "ignore",
                "target_rotation": "z",
            },
        },
        "touch_variants": [
            "",
            "RotateParallel",
            "RotateXYZ",
            "RotateZ",
        ],
    },
    "Egg": {
        "xml_file": "hand/manipulate_egg.xml",
        "touch_xml_file": "hand/manipulate_egg_touch_sensors.xml",
        "distance_threshold": 0.01,
        "randomize_initial_rotation": True,
        "ignore_z_target_rotation": False,
        "variants": {
            "": {
                "target_position": "random",
                "target_rotation": "xyz",
            },
            "Full": {
                "target_position": "random",
                "target_rotation": "xyz",
            },
            "Rotate": {
                "target_position": "ignore",
                "target_rotation": "xyz",
            },
        },
        "touch_variants": [
            "",
            "Rotate",
        ],
    },
    "Pen": {
        "xml_file": "hand/manipulate_pen.xml",
        "touch_xml_file": "hand/manipulate_pen_touch_sensors.xml",
        "distance_threshold": 0.05,
        "randomize_initial_rotation": False,
        "ignore_z_target_rotation": True,
        "variants": {
            "": {
                "target_position": "random",
                "target_rotation": "xyz",
            },
            "Full": {
                "target_position": "random",
                "target_rotation": "xyz",
            },
            "Rotate": {
                "target_position": "ignore",
                "target_rotation": "xyz",
            },
        },
        "touch_variants": [
            "",
            "Rotate",
        ],
    },
}

_ADROIT_TASKS: dict[str, dict[str, Any]] = {
    "AdroitHandDoor": {
        "xml_file": "adroit_hand/adroit_door.xml",
        "adroit_task": "door",
        "obs_dim": 39,
        "action_dim": 28,
        "qpos_dim": 30,
        "qvel_dim": 30,
        "reset_dim": 3,
    },
    "AdroitHandHammer": {
        "xml_file": "adroit_hand/adroit_hammer.xml",
        "adroit_task": "hammer",
        "obs_dim": 46,
        "action_dim": 26,
        "qpos_dim": 33,
        "qvel_dim": 33,
        "reset_dim": 3,
    },
    "AdroitHandPen": {
        "xml_file": "adroit_hand/adroit_pen.xml",
        "adroit_task": "pen",
        "obs_dim": 45,
        "action_dim": 24,
        "qpos_dim": 30,
        "qvel_dim": 30,
        "reset_dim": 4,
    },
    "AdroitHandRelocate": {
        "xml_file": "adroit_hand/adroit_relocate.xml",
        "adroit_task": "relocate",
        "obs_dim": 39,
        "action_dim": 30,
        "qpos_dim": 36,
        "qvel_dim": 36,
        "reset_dim": 6,
    },
}

_POINT_MAZE_TASKS: dict[str, dict[str, Any]] = {
    "PointMaze_Open": {
        "maze_map": "OPEN",
        "max_episode_steps": 300,
    },
    "PointMaze_UMaze": {
        "maze_map": "U_MAZE",
        "max_episode_steps": 300,
    },
    "PointMaze_Medium": {
        "maze_map": "MEDIUM_MAZE",
        "max_episode_steps": 600,
    },
    "PointMaze_Large": {
        "maze_map": "LARGE_MAZE",
        "max_episode_steps": 800,
    },
    "PointMaze_Open_Diverse_G": {
        "maze_map": "OPEN_DIVERSE_G",
        "max_episode_steps": 300,
    },
    "PointMaze_Open_Diverse_GR": {
        "maze_map": "OPEN_DIVERSE_GR",
        "max_episode_steps": 300,
    },
    "PointMaze_Medium_Diverse_G": {
        "maze_map": "MEDIUM_MAZE_DIVERSE_G",
        "max_episode_steps": 600,
    },
    "PointMaze_Medium_Diverse_GR": {
        "maze_map": "MEDIUM_MAZE_DIVERSE_GR",
        "max_episode_steps": 600,
    },
    "PointMaze_Large_Diverse_G": {
        "maze_map": "LARGE_MAZE_DIVERSE_G",
        "max_episode_steps": 800,
    },
    "PointMaze_Large_Diverse_GR": {
        "maze_map": "LARGE_MAZE_DIVERSE_GR",
        "max_episode_steps": 800,
    },
}

_KITCHEN_TASKS_TO_COMPLETE = [
    "bottom burner",
    "top burner",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
]

gymnasium_robotics_fetch_envs = [
    f"{task}{reward_suffix}-{version}"
    for task in _FETCH_TASKS
    for reward_suffix in ["", "Dense"]
    for version in ["v1", "v4"]
]

gymnasium_robotics_hand_envs = [
    "HandReach-v0",
    "HandReach-v3",
    "HandReachDense-v0",
    "HandReachDense-v3",
]

for obj_name, obj_conf in _HAND_MANIPULATE_OBJECTS.items():
    for variant in obj_conf["variants"]:
        for reward_suffix in ["", "Dense"]:
            for version in ["v0", "v1"]:
                gymnasium_robotics_hand_envs.append(
                    f"HandManipulate{obj_name}{variant}{reward_suffix}-{version}"
                )
    for variant in obj_conf["touch_variants"]:
        for touch_suffix in [
            "_BooleanTouchSensors",
            "_ContinuousTouchSensors",
        ]:
            for reward_suffix in ["", "Dense"]:
                for version in ["v0", "v1"]:
                    gymnasium_robotics_hand_envs.append(
                        f"HandManipulate{obj_name}{variant}"
                        f"{touch_suffix}{reward_suffix}-{version}"
                    )

gymnasium_robotics_adroit_envs = [
    f"{task}{reward_suffix}-v1"
    for task in _ADROIT_TASKS
    for reward_suffix in ["", "Sparse"]
]

gymnasium_robotics_point_maze_envs = [
    f"{task}{reward_suffix}-v3"
    for task in _POINT_MAZE_TASKS
    for reward_suffix in ["", "Dense"]
]

gymnasium_robotics_kitchen_envs = ["FrankaKitchen-v1"]


def _register_robotics_env(
    family: str,
    task_id: str,
    **kwargs: Any,
) -> None:
    spec_cls, dm_cls, gymnasium_cls = _ROBOTICS_ENV_CLASSES[family]
    register(
        task_id=task_id,
        import_path=_IMPORT_PATH,
        spec_cls=spec_cls,
        dm_cls=dm_cls,
        gymnasium_cls=gymnasium_cls,
        **kwargs,
    )


for task, kwargs in _FETCH_TASKS.items():
    for reward_suffix, reward_type in [("", "sparse"), ("Dense", "dense")]:
        for version in ["v1", "v4"]:
            _register_robotics_env(
                "Fetch",
                task_id=f"{task}{reward_suffix}-{version}",
                max_episode_steps=50,
                reward_type=reward_type,
                **kwargs,
            )

for reward_suffix, reward_type in [("", "sparse"), ("Dense", "dense")]:
    for version in ["v0", "v3"]:
        _register_robotics_env(
            "Hand",
            task_id=f"HandReach{reward_suffix}-{version}",
            max_episode_steps=50,
            xml_file="hand/reach.xml",
            hand_task="reach",
            reward_type=reward_type,
            obs_dim=63,
            goal_dim=15,
            qpos_dim=24,
            qvel_dim=24,
            relative_control=False,
            distance_threshold=0.01,
            rotation_threshold=0.1,
            target_position="ignore",
            target_rotation="ignore",
            randomize_initial_position=False,
            randomize_initial_rotation=False,
            ignore_z_target_rotation=False,
            touch_visualisation="off",
            touch_get_obs="off",
        )

for obj_name, obj_conf in _HAND_MANIPULATE_OBJECTS.items():
    for variant, variant_conf in obj_conf["variants"].items():
        for reward_suffix, reward_type in [("", "sparse"), ("Dense", "dense")]:
            for version in ["v0", "v1"]:
                _register_robotics_env(
                    "Hand",
                    task_id=(
                        f"HandManipulate{obj_name}{variant}{reward_suffix}-{version}"
                    ),
                    max_episode_steps=100,
                    xml_file=obj_conf["xml_file"],
                    hand_task="manipulate",
                    reward_type=reward_type,
                    obs_dim=61,
                    goal_dim=7,
                    qpos_dim=38,
                    qvel_dim=36,
                    relative_control=False,
                    distance_threshold=obj_conf["distance_threshold"],
                    rotation_threshold=0.1,
                    target_position=variant_conf["target_position"],
                    target_rotation=variant_conf["target_rotation"],
                    randomize_initial_position=True,
                    randomize_initial_rotation=obj_conf[
                        "randomize_initial_rotation"
                    ],
                    ignore_z_target_rotation=obj_conf[
                        "ignore_z_target_rotation"
                    ],
                    touch_visualisation="off",
                    touch_get_obs="off",
                )
    for variant in obj_conf["touch_variants"]:
        variant_conf = obj_conf["variants"][variant]
        for touch_suffix, touch_get_obs in [
            ("_BooleanTouchSensors", "boolean"),
            ("_ContinuousTouchSensors", "sensordata"),
        ]:
            for reward_suffix, reward_type in [
                ("", "sparse"),
                ("Dense", "dense"),
            ]:
                for version in ["v0", "v1"]:
                    _register_robotics_env(
                        "Hand",
                        task_id=(
                            f"HandManipulate{obj_name}{variant}"
                            f"{touch_suffix}{reward_suffix}-{version}"
                        ),
                        max_episode_steps=100,
                        xml_file=obj_conf["touch_xml_file"],
                        hand_task="manipulate",
                        reward_type=reward_type,
                        obs_dim=153,
                        goal_dim=7,
                        qpos_dim=38,
                        qvel_dim=36,
                        relative_control=False,
                        distance_threshold=obj_conf["distance_threshold"],
                        rotation_threshold=0.1,
                        target_position=variant_conf["target_position"],
                        target_rotation=variant_conf["target_rotation"],
                        randomize_initial_position=True,
                        randomize_initial_rotation=obj_conf[
                            "randomize_initial_rotation"
                        ],
                        ignore_z_target_rotation=obj_conf[
                            "ignore_z_target_rotation"
                        ],
                        touch_visualisation="on_touch",
                        touch_get_obs=touch_get_obs,
                    )

for task, kwargs in _ADROIT_TASKS.items():
    for reward_suffix, reward_type in [("", "dense"), ("Sparse", "sparse")]:
        _register_robotics_env(
            "Adroit",
            task_id=f"{task}{reward_suffix}-v1",
            max_episode_steps=200,
            reward_type=reward_type,
            **kwargs,
        )

for task, kwargs in _POINT_MAZE_TASKS.items():
    for reward_suffix, reward_type in [("", "sparse"), ("Dense", "dense")]:
        _register_robotics_env(
            "PointMaze",
            task_id=f"{task}{reward_suffix}-v3",
            reward_type=reward_type,
            continuing_task=True,
            reset_target=False,
            maze_size_scaling=1.0,
            maze_height=0.4,
            position_noise_range=0.25,
            **kwargs,
        )

_register_robotics_env(
    "Kitchen",
    task_id="FrankaKitchen-v1",
    max_episode_steps=280,
    frame_skip=40,
    xml_file="kitchen_franka/kitchen_assets/kitchen_env_model.xml",
    tasks_to_complete=_KITCHEN_TASKS_TO_COMPLETE,
    terminate_on_tasks_completed=True,
    remove_task_when_completed=True,
    robot_noise_ratio=0.01,
    object_noise_ratio=0.0005,
)
