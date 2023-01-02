# Copyright 2023 Garena Online Private Limited
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
"""Procgen env registration."""
from envpool.registration import register

# in total 16 games in Procgen
# 16 games has different timeout settings on their own
procgen_timeout_list = {
  "bigfish": 6000,
  "bossfight": 4000,
  "caveflyer": 1000,
  "chaser": 1000,
  "climber": 1000,
  "coinrun": 1000,
  "dodgeball": 1000,
  "fruitbot": 1000,
  "heist": 1000,
  "jumper": 1000,
  "leaper": 500,
  "maze": 500,
  "miner": 1000,
  "ninja": 1000,
  "plunder": 4000,
  "starpilot": 1000,
}

distribution_name = ["Easy", "Hard"]

distribution_code = [0, 1]

for env_name, timeout in procgen_timeout_list.items():
  for dist_name in distribution_name:
    for dist_code in distribution_code:
      register(
        task_id=f"{env_name.capitalize()}{dist_name}-v0",
        import_path="envpool.procgen",
        spec_cls="ProcgenEnvSpec",
        dm_cls="ProcgenDMEnvPool",
        gym_cls="ProcgenGymEnvPool",
        gymnasium_cls="ProcgenGymnasiumEnvPool",
        env_name=env_name,
        distribution_mode=dist_code,
        max_episode_steps=timeout,
      )
