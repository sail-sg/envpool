# Copyright 2022 Garena Online Private Limited
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
"""Mujoco gym env registration."""

from envpool.registration import register

gym_mujoco_envs = [
  ("Ant", "v3", False, 1000),
  ("Ant", "v5", True, 1000),
  ("HalfCheetah", "v3", False, 1000),
  ("HalfCheetah", "v5", True, 1000),
  ("Hopper", "v3", False, 1000),
  ("Hopper", "v5", True, 1000),
  ("Humanoid", "v3", False, 1000),
  ("Humanoid", "v5", True, 1000),
  ("HumanoidStandup", "v2", False, 1000),
  ("HumanoidStandup", "v5", True, 1000),
  ("InvertedDoublePendulum", "v2", False, 1000),
  ("InvertedDoublePendulum", "v5", True, 1000),
  ("InvertedPendulum", "v2", False, 1000),
  ("InvertedPendulum", "v5", True, 1000),
  ("Pusher", "v2", False, 100),
  ("Pusher", "v5", True, 100),
  ("Reacher", "v2", False, 50),
  ("Reacher", "v5", True, 50),
  ("Swimmer", "v3", False, 1000),
  ("Swimmer", "v5", True, 1000),
  ("Walker2d", "v3", False, 1000),
  ("Walker2d", "v5", True, 1000),
]

for task, version, post_constraint, max_episode_steps in gym_mujoco_envs:
  extra_args = {}
  if task in ["Ant", "Humanoid"] and version == "v3":
    extra_args["use_contact_force"] = True
  if task == "Ant" and version == "v5":
    extra_args.update(
      {
        "use_contact_force": True,
        "legacy_healthy_reward": False,
        "exclude_worldbody_contact_forces": True,
      }
    )
  if task == "Hopper" and version == "v5":
    extra_args["legacy_healthy_reward"] = False
  if task == "Humanoid" and version == "v5":
    extra_args.update(
      {
        "use_contact_force": True,
        "legacy_healthy_reward": False,
        "exclude_worldbody_observations": True,
        "exclude_root_actuator_forces": True,
      }
    )
  if task == "HumanoidStandup" and version == "v5":
    extra_args.update(
      {
        "exclude_worldbody_observations": True,
        "exclude_root_actuator_forces": True,
      }
    )
  if task == "InvertedDoublePendulum" and version == "v5":
    extra_args.update(
      {
        "constraint_obs_dim": 1,
        "reward_if_not_terminated": True,
      }
    )
  if task == "InvertedPendulum" and version == "v5":
    extra_args["reward_if_not_terminated"] = True
  if task == "Pusher" and version == "v5":
    extra_args.update(
      {
        "xml_file": "pusher_v5.xml",
        "reward_after_step": True,
        "weighted_reward_info": True,
      }
    )
  if task == "Reacher" and version == "v5":
    extra_args.update(
      {
        "reward_after_step": True,
        "obs_include_z_distance": False,
      }
    )
  if task == "Walker2d" and version == "v5":
    extra_args.update(
      {
        "xml_file": "walker2d_v5.xml",
        "legacy_healthy_reward": False,
      }
    )
  register(
    task_id=f"{task}-{version}",
    import_path="envpool.mujoco.gym",
    spec_cls=f"Gym{task}EnvSpec",
    dm_cls=f"Gym{task}DMEnvPool",
    gym_cls=f"Gym{task}GymEnvPool",
    gymnasium_cls=f"Gym{task}GymnasiumEnvPool",
    post_constraint=post_constraint,
    max_episode_steps=max_episode_steps,
    **extra_args,
  )
