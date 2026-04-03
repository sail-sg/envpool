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

from __future__ import annotations

import gymnasium
import gymnasium_robotics  # noqa: F401

from envpool.registration import register, registry

_IMPORT_PATH = "envpool.gymnasium_robotics"
_SPEC_CLS = "GymnasiumRoboticsEnvSpec"
_DM_CLS = "GymnasiumRoboticsDMEnvPool"
_GYM_CLS = "GymnasiumRoboticsGymEnvPool"
_GYMNASIUM_CLS = "GymnasiumRoboticsGymnasiumEnvPool"

_MUJOCO_SKIP_TASKS = {
    "Ant-v2",
    "Ant-v3",
    "HalfCheetah-v2",
    "HalfCheetah-v3",
    "Hopper-v2",
    "Hopper-v3",
    "Humanoid-v2",
    "Humanoid-v3",
    "HumanoidStandup-v2",
    "InvertedDoublePendulum-v2",
    "InvertedPendulum-v2",
    "Pusher-v2",
    "Reacher-v2",
    "Swimmer-v2",
    "Swimmer-v3",
    "Walker2d-v2",
    "Walker2d-v3",
}


def _upstream_entry_point(task_id: str) -> str:
    spec = gymnasium.envs.registry[task_id]
    return str(spec.entry_point)


def _gymnasium_task_id(task_id: str) -> str:
    entry_point = _upstream_entry_point(task_id)
    if task_id.startswith("Fetch") and entry_point.endswith(
        "MujocoPyFetchPickAndPlaceEnv"
    ):
        return task_id.replace("-v1", "-v4")
    if task_id.startswith("Fetch") and entry_point.endswith(
        "MujocoPyFetchPushEnv"
    ):
        return task_id.replace("-v1", "-v4")
    if task_id.startswith("Fetch") and entry_point.endswith(
        "MujocoPyFetchReachEnv"
    ):
        return task_id.replace("-v1", "-v4")
    if task_id.startswith("Fetch") and entry_point.endswith(
        "MujocoPyFetchSlideEnv"
    ):
        return task_id.replace("-v1", "-v4")
    if task_id.startswith("HandReach") and task_id.endswith("-v0"):
        return task_id.replace("-v0", "-v3")
    if task_id.startswith("HandManipulate") and task_id.endswith("-v0"):
        return task_id.replace("-v0", "-v1")
    return task_id


gymnasium_robotics_envs = [
    task_id
    for task_id, spec in sorted(gymnasium.envs.registry.items())
    if "gymnasium_robotics" in str(spec.entry_point)
    and task_id not in _MUJOCO_SKIP_TASKS
]

for task_id in gymnasium_robotics_envs:
    if task_id in registry.specs:
        continue
    upstream_spec = gymnasium.envs.registry[task_id]
    register(
        task_id=task_id,
        import_path=_IMPORT_PATH,
        spec_cls=_SPEC_CLS,
        dm_cls=_DM_CLS,
        gym_cls=_GYM_CLS,
        gymnasium_cls=_GYMNASIUM_CLS,
        gymnasium_task_id=_gymnasium_task_id(task_id),
        max_episode_steps=upstream_spec.max_episode_steps or 0,
        reward_threshold=upstream_spec.reward_threshold,
        gymnasium_robotics_kwargs=dict(upstream_spec.kwargs or {}),
    )
