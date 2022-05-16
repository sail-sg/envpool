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
"""Mujoco env registration."""

import os

from envpool.registration import register

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# from suite.BENCHMARKING
dmc_mujoco_envs = [
  ("cheetah", "run"),
  ("finger", "spin"),
  ("finger", "turn_easy"),
  ("finger", "turn_hard"),
  ("hopper", "hop"),
  ("hopper", "stand"),
  ("reacher", "easy"),
  ("reacher", "hard"),
  ("walker", "run"),
  ("walker", "stand"),
  ("walker", "walk"),
]

gym_mujoco_envs = [
  ("Ant", "v3", False),
  ("Ant", "v4", True),
  ("HalfCheetah", "v3", False),
  ("HalfCheetah", "v4", True),
  ("Hopper", "v3", False),
  ("Hopper", "v4", True),
  ("Humanoid", "v3", False),
  ("Humanoid", "v4", True),
  ("HumanoidStandup", "v2", False),
  ("HumanoidStandup", "v4", True),
  ("InvertedDoublePendulum", "v2", False),
  ("InvertedDoublePendulum", "v4", True),
  ("InvertedPendulum", "v2", False),
  ("InvertedPendulum", "v4", True),
  ("Pusher", "v2", False),
  ("Pusher", "v4", True),
  ("Reacher", "v2", False),
  ("Reacher", "v4", True),
  ("Swimmer", "v3", False),
  ("Swimmer", "v4", True),
  ("Walker2d", "v3", False),
  ("Walker2d", "v4", True),
]

for domain, task in dmc_mujoco_envs:
  domain_name = "".join([g.capitalize() for g in domain.split("_")])
  task_name = "".join([g.capitalize() for g in task.split("_")])
  register(
    task_id=f"{domain_name}{task_name}-v1",
    import_path="envpool.mujoco",
    spec_cls=f"Dmc{domain_name}EnvSpec",
    dm_cls=f"Dmc{domain_name}DMEnvPool",
    gym_cls=f"Dmc{domain_name}GymEnvPool",
    base_path=base_path,
    task_name=task,
  )

for task, version, post_constraint in gym_mujoco_envs:
  register(
    task_id=f"{task}-{version}",
    import_path="envpool.mujoco",
    spec_cls=f"Gym{task}EnvSpec",
    dm_cls=f"Gym{task}DMEnvPool",
    gym_cls=f"Gym{task}GymEnvPool",
    base_path=base_path,
    post_constraint=post_constraint,
  )
