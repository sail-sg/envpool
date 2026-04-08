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

_OFFICIAL_COMMON = {
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
    spec_cls="PyOfficialKinematics7Action5EnvSpec",
    dm_cls="PyOfficialKinematics7Action5DMEnvPool",
    gym_cls="PyOfficialKinematics7Action5GymEnvPool",
    gymnasium_cls="PyOfficialKinematics7Action5GymnasiumEnvPool",
    official_env_id="exit-v0",
    max_episode_steps=90,
    **_OFFICIAL_COMMON,
)

register(
    task_id="Intersection-v0",
    aliases=["intersection-v0"],
    spec_cls="PyOfficialKinematics7Action3EnvSpec",
    dm_cls="PyOfficialKinematics7Action3DMEnvPool",
    gym_cls="PyOfficialKinematics7Action3GymEnvPool",
    gymnasium_cls="PyOfficialKinematics7Action3GymnasiumEnvPool",
    official_env_id="intersection-v0",
    max_episode_steps=13,
    **_OFFICIAL_COMMON,
)

register(
    task_id="Intersection-v1",
    aliases=["intersection-v1"],
    spec_cls="PyOfficialKinematics8ContinuousEnvSpec",
    dm_cls="PyOfficialKinematics8ContinuousDMEnvPool",
    gym_cls="PyOfficialKinematics8ContinuousGymEnvPool",
    gymnasium_cls="PyOfficialKinematics8ContinuousGymnasiumEnvPool",
    official_env_id="intersection-v1",
    max_episode_steps=13,
    **_OFFICIAL_COMMON,
)

register(
    task_id="IntersectionMultiAgent-v0",
    aliases=["intersection-multi-agent-v0"],
    spec_cls="PyOfficialMultiAgentEnvSpec",
    dm_cls="PyOfficialMultiAgentDMEnvPool",
    gym_cls="PyOfficialMultiAgentGymEnvPool",
    gymnasium_cls="PyOfficialMultiAgentGymnasiumEnvPool",
    official_env_id="intersection-multi-agent-v0",
    max_episode_steps=13,
    max_num_players=2,
    **_OFFICIAL_COMMON,
)

register(
    task_id="IntersectionMultiAgent-v1",
    aliases=["intersection-multi-agent-v1"],
    spec_cls="PyOfficialMultiAgentEnvSpec",
    dm_cls="PyOfficialMultiAgentDMEnvPool",
    gym_cls="PyOfficialMultiAgentGymEnvPool",
    gymnasium_cls="PyOfficialMultiAgentGymnasiumEnvPool",
    official_env_id="intersection-multi-agent-v1",
    max_episode_steps=13,
    max_num_players=2,
    **_OFFICIAL_COMMON,
)

register(
    task_id="LaneKeeping-v0",
    aliases=["lane-keeping-v0"],
    spec_cls="PyOfficialAttributesEnvSpec",
    dm_cls="PyOfficialAttributesDMEnvPool",
    gym_cls="PyOfficialAttributesGymEnvPool",
    gymnasium_cls="PyOfficialAttributesGymnasiumEnvPool",
    official_env_id="lane-keeping-v0",
    max_episode_steps=200,
    **_OFFICIAL_COMMON,
)

register(
    task_id="Merge-v0",
    aliases=["merge-v0"],
    spec_cls="PyOfficialKinematics5EnvSpec",
    dm_cls="PyOfficialKinematics5DMEnvPool",
    gym_cls="PyOfficialKinematics5GymEnvPool",
    gymnasium_cls="PyOfficialKinematics5GymnasiumEnvPool",
    official_env_id="merge-v0",
    max_episode_steps=200,
    **_OFFICIAL_COMMON,
)

register(
    task_id="Parking-v0",
    aliases=["parking-v0"],
    spec_cls="PyOfficialGoalEnvSpec",
    dm_cls="PyOfficialGoalDMEnvPool",
    gym_cls="PyOfficialGoalGymEnvPool",
    gymnasium_cls="PyOfficialGoalGymnasiumEnvPool",
    official_env_id="parking-v0",
    max_episode_steps=500,
    **_OFFICIAL_COMMON,
)

register(
    task_id="ParkingActionRepeat-v0",
    aliases=["parking-ActionRepeat-v0"],
    spec_cls="PyOfficialGoalEnvSpec",
    dm_cls="PyOfficialGoalDMEnvPool",
    gym_cls="PyOfficialGoalGymEnvPool",
    gymnasium_cls="PyOfficialGoalGymnasiumEnvPool",
    official_env_id="parking-ActionRepeat-v0",
    max_episode_steps=500,
    **_OFFICIAL_COMMON,
)

register(
    task_id="ParkingParked-v0",
    aliases=["parking-parked-v0"],
    spec_cls="PyOfficialGoalEnvSpec",
    dm_cls="PyOfficialGoalDMEnvPool",
    gym_cls="PyOfficialGoalGymEnvPool",
    gymnasium_cls="PyOfficialGoalGymnasiumEnvPool",
    official_env_id="parking-parked-v0",
    max_episode_steps=500,
    **_OFFICIAL_COMMON,
)

register(
    task_id="Racetrack-v0",
    aliases=["racetrack-v0"],
    spec_cls="PyOfficialOccupancyEnvSpec",
    dm_cls="PyOfficialOccupancyDMEnvPool",
    gym_cls="PyOfficialOccupancyGymEnvPool",
    gymnasium_cls="PyOfficialOccupancyGymnasiumEnvPool",
    official_env_id="racetrack-v0",
    max_episode_steps=1500,
    **_OFFICIAL_COMMON,
)

register(
    task_id="RacetrackLarge-v0",
    aliases=["racetrack-large-v0"],
    spec_cls="PyOfficialOccupancyEnvSpec",
    dm_cls="PyOfficialOccupancyDMEnvPool",
    gym_cls="PyOfficialOccupancyGymEnvPool",
    gymnasium_cls="PyOfficialOccupancyGymnasiumEnvPool",
    official_env_id="racetrack-large-v0",
    max_episode_steps=1500,
    **_OFFICIAL_COMMON,
)

register(
    task_id="RacetrackOval-v0",
    aliases=["racetrack-oval-v0"],
    spec_cls="PyOfficialOccupancyEnvSpec",
    dm_cls="PyOfficialOccupancyDMEnvPool",
    gym_cls="PyOfficialOccupancyGymEnvPool",
    gymnasium_cls="PyOfficialOccupancyGymnasiumEnvPool",
    official_env_id="racetrack-oval-v0",
    max_episode_steps=1500,
    **_OFFICIAL_COMMON,
)

register(
    task_id="Roundabout-v0",
    aliases=["roundabout-v0"],
    spec_cls="PyOfficialKinematics5EnvSpec",
    dm_cls="PyOfficialKinematics5DMEnvPool",
    gym_cls="PyOfficialKinematics5GymEnvPool",
    gymnasium_cls="PyOfficialKinematics5GymnasiumEnvPool",
    official_env_id="roundabout-v0",
    max_episode_steps=11,
    **_OFFICIAL_COMMON,
)

register(
    task_id="TwoWay-v0",
    aliases=["two-way-v0"],
    spec_cls="PyOfficialTTC5EnvSpec",
    dm_cls="PyOfficialTTC5DMEnvPool",
    gym_cls="PyOfficialTTC5GymEnvPool",
    gymnasium_cls="PyOfficialTTC5GymnasiumEnvPool",
    official_env_id="two-way-v0",
    max_episode_steps=15,
    **_OFFICIAL_COMMON,
)

register(
    task_id="UTurn-v0",
    aliases=["u-turn-v0"],
    spec_cls="PyOfficialTTC16EnvSpec",
    dm_cls="PyOfficialTTC16DMEnvPool",
    gym_cls="PyOfficialTTC16GymEnvPool",
    gymnasium_cls="PyOfficialTTC16GymnasiumEnvPool",
    official_env_id="u-turn-v0",
    max_episode_steps=10,
    **_OFFICIAL_COMMON,
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
