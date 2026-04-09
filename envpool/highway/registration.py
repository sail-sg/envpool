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
"""Highway env registration."""

from envpool.registration import register

_COMMON = {
    "import_path": "envpool.highway",
    "spec_cls": "HighwayEnvSpec",
    "dm_cls": "HighwayDMEnvPool",
    "gym_cls": "HighwayGymEnvPool",
    "gymnasium_cls": "HighwayGymnasiumEnvPool",
}

_NATIVE_COMMON = {
    "import_path": "envpool.highway",
}

register(
    task_id="Highway-v0",
    aliases=["highway-v0"],
    max_episode_steps=40,
    duration=40,
    **_COMMON,
)

register(
    task_id="Exit-v0",
    aliases=["exit-v0"],
    spec_cls="NativeKinematics7Action5EnvSpec",
    dm_cls="NativeKinematics7Action5DMEnvPool",
    gym_cls="NativeKinematics7Action5GymEnvPool",
    gymnasium_cls="NativeKinematics7Action5GymnasiumEnvPool",
    scenario="exit",
    max_episode_steps=90,
    duration=90,
    simulation_frequency=5,
    screen_width=600,
    screen_height=150,
    **_NATIVE_COMMON,
)

register(
    task_id="Intersection-v0",
    aliases=["intersection-v0"],
    spec_cls="NativeKinematics7Action3EnvSpec",
    dm_cls="NativeKinematics7Action3DMEnvPool",
    gym_cls="NativeKinematics7Action3GymEnvPool",
    gymnasium_cls="NativeKinematics7Action3GymnasiumEnvPool",
    scenario="intersection",
    max_episode_steps=13,
    duration=13,
    screen_width=600,
    screen_height=600,
    **_NATIVE_COMMON,
)

register(
    task_id="Intersection-v1",
    aliases=["intersection-v1"],
    spec_cls="NativeKinematics8ContinuousEnvSpec",
    dm_cls="NativeKinematics8ContinuousDMEnvPool",
    gym_cls="NativeKinematics8ContinuousGymEnvPool",
    gymnasium_cls="NativeKinematics8ContinuousGymnasiumEnvPool",
    scenario="intersection_continuous",
    max_episode_steps=13,
    duration=13,
    screen_width=600,
    screen_height=600,
    **_NATIVE_COMMON,
)

register(
    task_id="IntersectionMultiAgent-v0",
    aliases=["intersection-multi-agent-v0"],
    spec_cls="NativeMultiAgentEnvSpec",
    dm_cls="NativeMultiAgentDMEnvPool",
    gym_cls="NativeMultiAgentGymEnvPool",
    gymnasium_cls="NativeMultiAgentGymnasiumEnvPool",
    scenario="intersection_multi",
    max_episode_steps=13,
    duration=13,
    screen_width=600,
    screen_height=600,
    max_num_players=2,
    **_NATIVE_COMMON,
)

register(
    task_id="IntersectionMultiAgent-v1",
    aliases=["intersection-multi-agent-v1"],
    spec_cls="NativeMultiAgentEnvSpec",
    dm_cls="NativeMultiAgentDMEnvPool",
    gym_cls="NativeMultiAgentGymEnvPool",
    gymnasium_cls="NativeMultiAgentGymnasiumEnvPool",
    scenario="intersection_multi",
    max_episode_steps=13,
    duration=13,
    screen_width=600,
    screen_height=600,
    max_num_players=2,
    **_NATIVE_COMMON,
)

register(
    task_id="LaneKeeping-v0",
    aliases=["lane-keeping-v0"],
    spec_cls="NativeAttributesEnvSpec",
    dm_cls="NativeAttributesDMEnvPool",
    gym_cls="NativeAttributesGymEnvPool",
    gymnasium_cls="NativeAttributesGymnasiumEnvPool",
    scenario="lane_keeping",
    max_episode_steps=200,
    duration=20,
    policy_frequency=10,
    screen_width=600,
    screen_height=250,
    **_NATIVE_COMMON,
)

register(
    task_id="Merge-v0",
    aliases=["merge-v0"],
    spec_cls="NativeKinematics5EnvSpec",
    dm_cls="NativeKinematics5DMEnvPool",
    gym_cls="NativeKinematics5GymEnvPool",
    gymnasium_cls="NativeKinematics5GymnasiumEnvPool",
    scenario="merge",
    max_episode_steps=200,
    duration=200,
    **_NATIVE_COMMON,
)

register(
    task_id="Parking-v0",
    aliases=["parking-v0"],
    spec_cls="NativeGoalEnvSpec",
    dm_cls="NativeGoalDMEnvPool",
    gym_cls="NativeGoalGymEnvPool",
    gymnasium_cls="NativeGoalGymnasiumEnvPool",
    scenario="parking",
    max_episode_steps=500,
    duration=100,
    policy_frequency=5,
    screen_width=600,
    screen_height=300,
    **_NATIVE_COMMON,
)

register(
    task_id="ParkingActionRepeat-v0",
    aliases=["parking-ActionRepeat-v0"],
    spec_cls="NativeGoalEnvSpec",
    dm_cls="NativeGoalDMEnvPool",
    gym_cls="NativeGoalGymEnvPool",
    gymnasium_cls="NativeGoalGymnasiumEnvPool",
    scenario="parking_action_repeat",
    max_episode_steps=500,
    duration=500,
    screen_width=600,
    screen_height=300,
    **_NATIVE_COMMON,
)

register(
    task_id="ParkingParked-v0",
    aliases=["parking-parked-v0"],
    spec_cls="NativeGoalEnvSpec",
    dm_cls="NativeGoalDMEnvPool",
    gym_cls="NativeGoalGymEnvPool",
    gymnasium_cls="NativeGoalGymnasiumEnvPool",
    scenario="parking_parked",
    max_episode_steps=500,
    duration=100,
    policy_frequency=5,
    screen_width=600,
    screen_height=300,
    **_NATIVE_COMMON,
)

register(
    task_id="Racetrack-v0",
    aliases=["racetrack-v0"],
    spec_cls="NativeOccupancyEnvSpec",
    dm_cls="NativeOccupancyDMEnvPool",
    gym_cls="NativeOccupancyGymEnvPool",
    gymnasium_cls="NativeOccupancyGymnasiumEnvPool",
    scenario="racetrack",
    max_episode_steps=1500,
    duration=300,
    policy_frequency=5,
    screen_width=600,
    screen_height=600,
    **_NATIVE_COMMON,
)

register(
    task_id="RacetrackLarge-v0",
    aliases=["racetrack-large-v0"],
    spec_cls="NativeOccupancyEnvSpec",
    dm_cls="NativeOccupancyDMEnvPool",
    gym_cls="NativeOccupancyGymEnvPool",
    gymnasium_cls="NativeOccupancyGymnasiumEnvPool",
    scenario="racetrack_large",
    max_episode_steps=1500,
    duration=300,
    policy_frequency=5,
    screen_width=600,
    screen_height=600,
    **_NATIVE_COMMON,
)

register(
    task_id="RacetrackOval-v0",
    aliases=["racetrack-oval-v0"],
    spec_cls="NativeOccupancyEnvSpec",
    dm_cls="NativeOccupancyDMEnvPool",
    gym_cls="NativeOccupancyGymEnvPool",
    gymnasium_cls="NativeOccupancyGymnasiumEnvPool",
    scenario="racetrack_oval",
    max_episode_steps=1500,
    duration=300,
    policy_frequency=5,
    screen_width=600,
    screen_height=600,
    **_NATIVE_COMMON,
)

register(
    task_id="Roundabout-v0",
    aliases=["roundabout-v0"],
    spec_cls="NativeKinematics5EnvSpec",
    dm_cls="NativeKinematics5DMEnvPool",
    gym_cls="NativeKinematics5GymEnvPool",
    gymnasium_cls="NativeKinematics5GymnasiumEnvPool",
    scenario="roundabout",
    max_episode_steps=11,
    duration=11,
    screen_width=600,
    screen_height=600,
    **_NATIVE_COMMON,
)

register(
    task_id="TwoWay-v0",
    aliases=["two-way-v0"],
    spec_cls="NativeTTC5EnvSpec",
    dm_cls="NativeTTC5DMEnvPool",
    gym_cls="NativeTTC5GymEnvPool",
    gymnasium_cls="NativeTTC5GymnasiumEnvPool",
    scenario="two_way",
    max_episode_steps=15,
    duration=15,
    **_NATIVE_COMMON,
)

register(
    task_id="UTurn-v0",
    aliases=["u-turn-v0"],
    spec_cls="NativeTTC16EnvSpec",
    dm_cls="NativeTTC16DMEnvPool",
    gym_cls="NativeTTC16GymEnvPool",
    gymnasium_cls="NativeTTC16GymnasiumEnvPool",
    scenario="u_turn",
    max_episode_steps=10,
    duration=10,
    screen_width=789,
    screen_height=289,
    **_NATIVE_COMMON,
)

register(
    task_id="HighwayFast-v0",
    aliases=["highway-fast-v0"],
    max_episode_steps=30,
    duration=30,
    simulation_frequency=5,
    lanes_count=3,
    vehicles_count=20,
    ego_spacing=1.5,
    other_vehicles_check_collisions=False,
    **_COMMON,
)
