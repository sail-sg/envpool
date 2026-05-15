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
"""MuJoCo Playground env registration."""

from envpool.registration import register

_IMPORT_PATH = "envpool.mujoco.playground"

_PLAYGROUND_TASKS = (
    ("AlohaHandOver", "PlaygroundAloha", 250, {}),
    (
        "AlohaSinglePegInsertion",
        "PlaygroundAloha",
        1000,
        {
            "ctrl_dt": 0.0025,
            "sim_dt": 0.0025,
            "action_repeat": 2,
            "action_scale": 0.005,
        },
    ),
    ("ApolloJoystickFlatTerrain", "PlaygroundApollo", 1000, {}),
    ("BarkourJoystick", "PlaygroundBarkour", 1000, {}),
    (
        "BerkeleyHumanoidJoystickFlatTerrain",
        "PlaygroundBerkeleyHumanoid",
        1000,
        {},
    ),
    (
        "BerkeleyHumanoidJoystickRoughTerrain",
        "PlaygroundBerkeleyHumanoid",
        1000,
        {},
    ),
    ("G1JoystickFlatTerrain", "PlaygroundG1", 1000, {}),
    ("G1JoystickRoughTerrain", "PlaygroundG1", 1000, {}),
    ("Go1JoystickFlatTerrain", "PlaygroundGo1", 1000, {}),
    ("Go1JoystickRoughTerrain", "PlaygroundGo1", 1000, {}),
    ("Go1Getup", "PlaygroundGo1Getup", 300, {}),
    ("Go1Handstand", "PlaygroundGo1Handstand", 500, {}),
    ("Go1Footstand", "PlaygroundGo1Handstand", 500, {}),
    (
        "H1InplaceGaitTracking",
        "PlaygroundH1",
        1000,
        {
            "action_scale": 0.6,
            "history_len": 3,
            "obs_noise_level": 1.0,
            "feet_phase_scale": 2.0,
            "pose_scale": -0.5,
            "gait_frequency_max": 4.0,
            "gait_count": 2,
        },
    ),
    ("H1JoystickGaitTracking", "PlaygroundH1", 1000, {}),
    (
        "LeapCubeReorient",
        "PlaygroundHand",
        1000,
        {
            "action_scale": 0.5,
            "success_reward": 100.0,
            "angvel_scale": 0.0,
            "orientation_scale": 5.0,
            "position_scale": 0.5,
            "hand_pose_scale": -0.5,
            "action_rate_scale": -0.001,
            "energy_scale": -0.001,
        },
    ),
    ("LeapCubeRotateZAxis", "PlaygroundHand", 500, {}),
    ("Op3Joystick", "PlaygroundOp3", 1000, {}),
    ("PandaPickCube", "PlaygroundPanda", 150, {}),
    (
        "PandaPickCubeCartesian",
        "PlaygroundPanda",
        200,
        {
            "ctrl_dt": 0.05,
            "sim_dt": 0.005,
            "action_scale": 0.005,
            "robot_target_qpos_scale": 0.0,
        },
    ),
    ("PandaPickCubeOrientation", "PlaygroundPanda", 150, {}),
    ("PandaOpenCabinet", "PlaygroundPanda", 150, {}),
    ("PandaRobotiqPushCube", "PlaygroundPandaRobotiq", 3000, {}),
    ("AeroCubeRotateZAxis", "PlaygroundHand", 500, {"action_rate_scale": -1.0}),
    ("SpotFlatTerrainJoystick", "PlaygroundSpotJoystick", 1000, {}),
    (
        "SpotGetup",
        "PlaygroundSpotGetup",
        300,
        {
            "kp": 400.0,
            "kd": 20.0,
            "action_scale": 0.6,
            "noise_joint_pos": 0.01,
            "noise_gyro": 0.2,
            "noise_gravity": 0.05,
            "orientation_scale": 1.0,
            "torso_height_scale": 1.0,
            "posture_scale": 1.0,
            "stand_still_scale": 1.0,
            "torques_scale": 0.0,
            "action_rate_scale": 0.0,
        },
    ),
    (
        "SpotJoystickGaitTracking",
        "PlaygroundSpotGait",
        1000,
        {
            "kp": 400.0,
            "kd": 10.0,
            "action_scale": 0.6,
            "tracking_lin_vel_scale": 0.5,
            "tracking_ang_vel_scale": 0.5,
            "feet_phase_scale": 2.0,
            "ang_vel_xy_scale": -0.5,
            "lin_vel_z_scale": -0.5,
            "hip_splay_scale": -0.5,
            "lin_vel_y_min": -0.5,
            "lin_vel_y_max": 0.5,
        },
    ),
    ("T1JoystickFlatTerrain", "PlaygroundT1", 1000, {}),
    ("T1JoystickRoughTerrain", "PlaygroundT1", 1000, {}),
)

_REGISTER_ORDER = (
    "AlohaHandOver",
    "AlohaSinglePegInsertion",
    "BarkourJoystick",
    "ApolloJoystickFlatTerrain",
    "Op3Joystick",
    "BerkeleyHumanoidJoystickFlatTerrain",
    "BerkeleyHumanoidJoystickRoughTerrain",
    "G1JoystickFlatTerrain",
    "G1JoystickRoughTerrain",
    "Go1JoystickFlatTerrain",
    "Go1JoystickRoughTerrain",
    "Go1Getup",
    "Go1Handstand",
    "Go1Footstand",
    "H1InplaceGaitTracking",
    "H1JoystickGaitTracking",
    "LeapCubeRotateZAxis",
    "LeapCubeReorient",
    "AeroCubeRotateZAxis",
    "PandaPickCube",
    "PandaPickCubeOrientation",
    "PandaOpenCabinet",
    "PandaPickCubeCartesian",
    "PandaRobotiqPushCube",
    "SpotFlatTerrainJoystick",
    "SpotGetup",
    "SpotJoystickGaitTracking",
    "T1JoystickFlatTerrain",
    "T1JoystickRoughTerrain",
)

PLAYGROUND_ENVS = tuple(task_name for task_name, _, _, _ in _PLAYGROUND_TASKS)
_TASK_BY_NAME = {task[0]: task for task in _PLAYGROUND_TASKS}


def _register_task(
    task_name: str,
    env_cls_prefix: str,
    max_episode_steps: int,
    kwargs: dict[str, object],
) -> None:
    register(
        task_id=f"{task_name}-v1",
        aliases=(f"MuJoCoPlayground/{task_name}-v1",),
        import_path=_IMPORT_PATH,
        spec_cls=f"{env_cls_prefix}EnvSpec",
        dm_cls=f"{env_cls_prefix}DMEnvPool",
        gymnasium_cls=f"{env_cls_prefix}GymnasiumEnvPool",
        task_name=task_name,
        max_episode_steps=max_episode_steps,
        **kwargs,
    )


for _task_name in _REGISTER_ORDER:
    _register_task(*_TASK_BY_NAME[_task_name])
