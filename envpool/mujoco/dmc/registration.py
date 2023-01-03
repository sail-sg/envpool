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

from envpool.registration import register

# from suite.BENCHMARKING
dmc_mujoco_envs = [
  ("acrobot", "swingup", 1000),
  ("acrobot", "swingup_sparse", 1000),
  ("ball_in_cup", "catch", 1000),
  ("cartpole", "balance", 1000),
  ("cartpole", "balance_sparse", 1000),
  ("cartpole", "swingup", 1000),
  ("cartpole", "swingup_sparse", 1000),
  ("cartpole", "three_poles", 1000),
  ("cartpole", "two_poles", 1000),
  ("cheetah", "run", 1000),
  ("finger", "spin", 1000),
  ("finger", "turn_easy", 1000),
  ("finger", "turn_hard", 1000),
  ("fish", "swim", 1000),
  ("fish", "upright", 1000),
  ("hopper", "hop", 1000),
  ("hopper", "stand", 1000),
  ("humanoid", "run", 1000),
  ("humanoid", "run_pure_state", 1000),
  ("humanoid", "stand", 1000),
  ("humanoid", "walk", 1000),
  ("humanoid_CMU", "run", 1000),
  ("humanoid_CMU", "stand", 1000),
  ("manipulator", "bring_ball", 1000),
  ("manipulator", "bring_peg", 1000),
  ("manipulator", "insert_ball", 1000),
  ("manipulator", "insert_peg", 1000),
  ("pendulum", "swingup", 1000),
  ("point_mass", "easy", 1000),
  ("point_mass", "hard", 1000),
  ("reacher", "easy", 1000),
  ("reacher", "hard", 1000),
  ("swimmer", "swimmer6", 1000),
  ("swimmer", "swimmer15", 1000),
  ("walker", "run", 1000),
  ("walker", "stand", 1000),
  ("walker", "walk", 1000),
]

for domain, task, max_episode_steps in dmc_mujoco_envs:
  domain_name = "".join([g[:1].upper() + g[1:] for g in domain.split("_")])
  task_name = "".join([g[:1].upper() + g[1:] for g in task.split("_")])
  register(
    task_id=f"{domain_name}{task_name}-v1",
    import_path="envpool.mujoco.dmc",
    spec_cls=f"Dmc{domain_name}EnvSpec",
    dm_cls=f"Dmc{domain_name}DMEnvPool",
    gym_cls=f"Dmc{domain_name}GymEnvPool",
    gymnasium_cls=f"Dmc{domain_name}GymnasiumEnvPool",
    task_name=task,
    max_episode_steps=max_episode_steps,
  )
