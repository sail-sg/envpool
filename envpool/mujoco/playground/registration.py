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

PLAYGROUND_ENVS = (
    "AlohaHandOver",
    "AlohaSinglePegInsertion",
    "ApolloJoystickFlatTerrain",
    "BarkourJoystick",
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
    "LeapCubeReorient",
    "LeapCubeRotateZAxis",
    "Op3Joystick",
    "PandaPickCube",
    "PandaPickCubeCartesian",
    "PandaPickCubeOrientation",
    "PandaOpenCabinet",
    "PandaRobotiqPushCube",
    "AeroCubeRotateZAxis",
    "SpotFlatTerrainJoystick",
    "SpotGetup",
    "SpotJoystickGaitTracking",
    "T1JoystickFlatTerrain",
    "T1JoystickRoughTerrain",
)

register(
    task_id="AlohaHandOver-v1",
    aliases=("MuJoCoPlayground/AlohaHandOver-v1",),
    import_path="envpool.mujoco.playground",
    spec_cls="PlaygroundAlohaEnvSpec",
    dm_cls="PlaygroundAlohaDMEnvPool",
    gymnasium_cls="PlaygroundAlohaGymnasiumEnvPool",
    task_name="AlohaHandOver",
    max_episode_steps=250,
)

register(
    task_id="AlohaSinglePegInsertion-v1",
    aliases=("MuJoCoPlayground/AlohaSinglePegInsertion-v1",),
    import_path="envpool.mujoco.playground",
    spec_cls="PlaygroundAlohaEnvSpec",
    dm_cls="PlaygroundAlohaDMEnvPool",
    gymnasium_cls="PlaygroundAlohaGymnasiumEnvPool",
    task_name="AlohaSinglePegInsertion",
    max_episode_steps=1000,
    ctrl_dt=0.0025,
    sim_dt=0.0025,
    action_repeat=2,
    action_scale=0.005,
)

register(
    task_id="BarkourJoystick-v1",
    aliases=("MuJoCoPlayground/BarkourJoystick-v1",),
    import_path="envpool.mujoco.playground",
    spec_cls="PlaygroundBarkourEnvSpec",
    dm_cls="PlaygroundBarkourDMEnvPool",
    gymnasium_cls="PlaygroundBarkourGymnasiumEnvPool",
    task_name="BarkourJoystick",
    max_episode_steps=1000,
)

register(
    task_id="ApolloJoystickFlatTerrain-v1",
    aliases=("MuJoCoPlayground/ApolloJoystickFlatTerrain-v1",),
    import_path="envpool.mujoco.playground",
    spec_cls="PlaygroundApolloEnvSpec",
    dm_cls="PlaygroundApolloDMEnvPool",
    gymnasium_cls="PlaygroundApolloGymnasiumEnvPool",
    task_name="ApolloJoystickFlatTerrain",
    max_episode_steps=1000,
)

register(
    task_id="Op3Joystick-v1",
    aliases=("MuJoCoPlayground/Op3Joystick-v1",),
    import_path="envpool.mujoco.playground",
    spec_cls="PlaygroundOp3EnvSpec",
    dm_cls="PlaygroundOp3DMEnvPool",
    gymnasium_cls="PlaygroundOp3GymnasiumEnvPool",
    task_name="Op3Joystick",
    max_episode_steps=1000,
)

_BERKELEY_HUMANOID_ENVS = (
    "BerkeleyHumanoidJoystickFlatTerrain",
    "BerkeleyHumanoidJoystickRoughTerrain",
)

for task_name in _BERKELEY_HUMANOID_ENVS:
    register(
        task_id=f"{task_name}-v1",
        aliases=(f"MuJoCoPlayground/{task_name}-v1",),
        import_path="envpool.mujoco.playground",
        spec_cls="PlaygroundBerkeleyHumanoidEnvSpec",
        dm_cls="PlaygroundBerkeleyHumanoidDMEnvPool",
        gymnasium_cls="PlaygroundBerkeleyHumanoidGymnasiumEnvPool",
        task_name=task_name,
        max_episode_steps=1000,
    )

_G1_ENVS = (
    "G1JoystickFlatTerrain",
    "G1JoystickRoughTerrain",
)

for task_name in _G1_ENVS:
    register(
        task_id=f"{task_name}-v1",
        aliases=(f"MuJoCoPlayground/{task_name}-v1",),
        import_path="envpool.mujoco.playground",
        spec_cls="PlaygroundG1EnvSpec",
        dm_cls="PlaygroundG1DMEnvPool",
        gymnasium_cls="PlaygroundG1GymnasiumEnvPool",
        task_name=task_name,
        max_episode_steps=1000,
    )

_JOYSTICK_ENVS = (
    "Go1JoystickFlatTerrain",
    "Go1JoystickRoughTerrain",
)

for task_name in _JOYSTICK_ENVS:
    register(
        task_id=f"{task_name}-v1",
        aliases=(f"MuJoCoPlayground/{task_name}-v1",),
        import_path="envpool.mujoco.playground",
        spec_cls="PlaygroundGo1EnvSpec",
        dm_cls="PlaygroundGo1DMEnvPool",
        gymnasium_cls="PlaygroundGo1GymnasiumEnvPool",
        task_name=task_name,
        max_episode_steps=1000,
    )

register(
    task_id="Go1Getup-v1",
    aliases=("MuJoCoPlayground/Go1Getup-v1",),
    import_path="envpool.mujoco.playground",
    spec_cls="PlaygroundGo1GetupEnvSpec",
    dm_cls="PlaygroundGo1GetupDMEnvPool",
    gymnasium_cls="PlaygroundGo1GetupGymnasiumEnvPool",
    task_name="Go1Getup",
    max_episode_steps=300,
)

register(
    task_id="Go1Handstand-v1",
    aliases=("MuJoCoPlayground/Go1Handstand-v1",),
    import_path="envpool.mujoco.playground",
    spec_cls="PlaygroundGo1HandstandEnvSpec",
    dm_cls="PlaygroundGo1HandstandDMEnvPool",
    gymnasium_cls="PlaygroundGo1HandstandGymnasiumEnvPool",
    task_name="Go1Handstand",
    max_episode_steps=500,
)

register(
    task_id="Go1Footstand-v1",
    aliases=("MuJoCoPlayground/Go1Footstand-v1",),
    import_path="envpool.mujoco.playground",
    spec_cls="PlaygroundGo1HandstandEnvSpec",
    dm_cls="PlaygroundGo1HandstandDMEnvPool",
    gymnasium_cls="PlaygroundGo1HandstandGymnasiumEnvPool",
    task_name="Go1Footstand",
    max_episode_steps=500,
)

register(
    task_id="H1InplaceGaitTracking-v1",
    aliases=("MuJoCoPlayground/H1InplaceGaitTracking-v1",),
    import_path="envpool.mujoco.playground",
    spec_cls="PlaygroundH1EnvSpec",
    dm_cls="PlaygroundH1DMEnvPool",
    gymnasium_cls="PlaygroundH1GymnasiumEnvPool",
    task_name="H1InplaceGaitTracking",
    max_episode_steps=1000,
    action_scale=0.6,
    history_len=3,
    obs_noise_level=1.0,
    feet_phase_scale=2.0,
    pose_scale=-0.5,
    gait_frequency_max=4.0,
    gait_count=2,
)

register(
    task_id="H1JoystickGaitTracking-v1",
    aliases=("MuJoCoPlayground/H1JoystickGaitTracking-v1",),
    import_path="envpool.mujoco.playground",
    spec_cls="PlaygroundH1EnvSpec",
    dm_cls="PlaygroundH1DMEnvPool",
    gymnasium_cls="PlaygroundH1GymnasiumEnvPool",
    task_name="H1JoystickGaitTracking",
    max_episode_steps=1000,
)

register(
    task_id="LeapCubeRotateZAxis-v1",
    aliases=("MuJoCoPlayground/LeapCubeRotateZAxis-v1",),
    import_path="envpool.mujoco.playground",
    spec_cls="PlaygroundHandEnvSpec",
    dm_cls="PlaygroundHandDMEnvPool",
    gymnasium_cls="PlaygroundHandGymnasiumEnvPool",
    task_name="LeapCubeRotateZAxis",
    max_episode_steps=500,
)

register(
    task_id="LeapCubeReorient-v1",
    aliases=("MuJoCoPlayground/LeapCubeReorient-v1",),
    import_path="envpool.mujoco.playground",
    spec_cls="PlaygroundHandEnvSpec",
    dm_cls="PlaygroundHandDMEnvPool",
    gymnasium_cls="PlaygroundHandGymnasiumEnvPool",
    task_name="LeapCubeReorient",
    max_episode_steps=1000,
    action_scale=0.5,
    success_reward=100.0,
    angvel_scale=0.0,
    orientation_scale=5.0,
    position_scale=0.5,
    hand_pose_scale=-0.5,
    action_rate_scale=-0.001,
    energy_scale=-0.001,
)

register(
    task_id="AeroCubeRotateZAxis-v1",
    aliases=("MuJoCoPlayground/AeroCubeRotateZAxis-v1",),
    import_path="envpool.mujoco.playground",
    spec_cls="PlaygroundHandEnvSpec",
    dm_cls="PlaygroundHandDMEnvPool",
    gymnasium_cls="PlaygroundHandGymnasiumEnvPool",
    task_name="AeroCubeRotateZAxis",
    max_episode_steps=500,
    action_rate_scale=-1.0,
)

_PANDA_ENVS = (
    "PandaPickCube",
    "PandaPickCubeOrientation",
    "PandaOpenCabinet",
)

for task_name in _PANDA_ENVS:
    register(
        task_id=f"{task_name}-v1",
        aliases=(f"MuJoCoPlayground/{task_name}-v1",),
        import_path="envpool.mujoco.playground",
        spec_cls="PlaygroundPandaEnvSpec",
        dm_cls="PlaygroundPandaDMEnvPool",
        gymnasium_cls="PlaygroundPandaGymnasiumEnvPool",
        task_name=task_name,
        max_episode_steps=150,
    )

register(
    task_id="PandaPickCubeCartesian-v1",
    aliases=("MuJoCoPlayground/PandaPickCubeCartesian-v1",),
    import_path="envpool.mujoco.playground",
    spec_cls="PlaygroundPandaEnvSpec",
    dm_cls="PlaygroundPandaDMEnvPool",
    gymnasium_cls="PlaygroundPandaGymnasiumEnvPool",
    task_name="PandaPickCubeCartesian",
    max_episode_steps=200,
    ctrl_dt=0.05,
    sim_dt=0.005,
    action_scale=0.005,
    robot_target_qpos_scale=0.0,
)

register(
    task_id="PandaRobotiqPushCube-v1",
    aliases=("MuJoCoPlayground/PandaRobotiqPushCube-v1",),
    import_path="envpool.mujoco.playground",
    spec_cls="PlaygroundPandaRobotiqEnvSpec",
    dm_cls="PlaygroundPandaRobotiqDMEnvPool",
    gymnasium_cls="PlaygroundPandaRobotiqGymnasiumEnvPool",
    task_name="PandaRobotiqPushCube",
    max_episode_steps=3000,
)

register(
    task_id="SpotFlatTerrainJoystick-v1",
    aliases=("MuJoCoPlayground/SpotFlatTerrainJoystick-v1",),
    import_path="envpool.mujoco.playground",
    spec_cls="PlaygroundSpotJoystickEnvSpec",
    dm_cls="PlaygroundSpotJoystickDMEnvPool",
    gymnasium_cls="PlaygroundSpotJoystickGymnasiumEnvPool",
    task_name="SpotFlatTerrainJoystick",
    max_episode_steps=1000,
)

register(
    task_id="SpotGetup-v1",
    aliases=("MuJoCoPlayground/SpotGetup-v1",),
    import_path="envpool.mujoco.playground",
    spec_cls="PlaygroundSpotGetupEnvSpec",
    dm_cls="PlaygroundSpotGetupDMEnvPool",
    gymnasium_cls="PlaygroundSpotGetupGymnasiumEnvPool",
    task_name="SpotGetup",
    max_episode_steps=300,
    kp=400.0,
    kd=20.0,
    action_scale=0.6,
    noise_joint_pos=0.01,
    noise_gyro=0.2,
    noise_gravity=0.05,
    orientation_scale=1.0,
    torso_height_scale=1.0,
    posture_scale=1.0,
    stand_still_scale=1.0,
    torques_scale=0.0,
    action_rate_scale=0.0,
)

register(
    task_id="SpotJoystickGaitTracking-v1",
    aliases=("MuJoCoPlayground/SpotJoystickGaitTracking-v1",),
    import_path="envpool.mujoco.playground",
    spec_cls="PlaygroundSpotGaitEnvSpec",
    dm_cls="PlaygroundSpotGaitDMEnvPool",
    gymnasium_cls="PlaygroundSpotGaitGymnasiumEnvPool",
    task_name="SpotJoystickGaitTracking",
    max_episode_steps=1000,
    kp=400.0,
    kd=10.0,
    action_scale=0.6,
    tracking_lin_vel_scale=0.5,
    tracking_ang_vel_scale=0.5,
    feet_phase_scale=2.0,
    ang_vel_xy_scale=-0.5,
    lin_vel_z_scale=-0.5,
    hip_splay_scale=-0.5,
    lin_vel_y_min=-0.5,
    lin_vel_y_max=0.5,
)

_T1_ENVS = (
    "T1JoystickFlatTerrain",
    "T1JoystickRoughTerrain",
)

for task_name in _T1_ENVS:
    register(
        task_id=f"{task_name}-v1",
        aliases=(f"MuJoCoPlayground/{task_name}-v1",),
        import_path="envpool.mujoco.playground",
        spec_cls="PlaygroundT1EnvSpec",
        dm_cls="PlaygroundT1DMEnvPool",
        gymnasium_cls="PlaygroundT1GymnasiumEnvPool",
        task_name=task_name,
        max_episode_steps=1000,
    )
