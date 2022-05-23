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
"""Mujoco dm_control suite env registration."""

import os

from envpool.registration import register

base_path = os.path.abspath(
  os.path.join(os.path.dirname(__file__), "..", "..")
)

# from suite.BENCHMARKING
dmc_mujoco_envs = [
  ("acrobot", "swingup"),
  ("acrobot", "swingup_sparse"),
  ("ball_in_cup", "catch"),
  ("cartpole", "balance"),
  ("cartpole", "balance_sparse"),
  ("cartpole", "swingup"),
  ("cartpole", "swingup_sparse"),
  ("cartpole", "three_poles"),
  ("cartpole", "two_poles"),
  ("cheetah", "run"),
  ("finger", "spin"),
  ("finger", "turn_easy"),
  ("finger", "turn_hard"),
  ("fish", "swim"),
  ("fish", "upright"),
  ("hopper", "hop"),
  ("hopper", "stand"),
  ("humanoid", "run"),
  ("humanoid", "run_pure_state"),
  ("humanoid", "stand"),
  ("humanoid", "walk"),
  ("manipulator", "bring_ball"),
  ("manipulator", "bring_peg"),
  ("manipulator", "insert_ball"),
  ("manipulator", "insert_peg"),
  ("pendulum", "swingup"),
  ("point_mass", "easy"),
  ("point_mass", "hard"),
  ("reacher", "easy"),
  ("reacher", "hard"),
  ("swimmer", "swimmer6"),
  ("swimmer", "swimmer15"),
  ("walker", "run"),
  ("walker", "stand"),
  ("walker", "walk"),
]

for domain, task in dmc_mujoco_envs:
  domain_name = "".join([g.capitalize() for g in domain.split("_")])
  task_name = "".join([g.capitalize() for g in task.split("_")])
  register(
    task_id=f"{domain_name}{task_name}-v1",
    import_path="envpool.mujoco.dmc",
    spec_cls=f"Dmc{domain_name}EnvSpec",
    dm_cls=f"Dmc{domain_name}DMEnvPool",
    gym_cls=f"Dmc{domain_name}GymEnvPool",
    base_path=base_path,
    task_name=task,
  )
