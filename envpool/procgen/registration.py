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

# 16 games in Procgen
# https://github.com/openai/procgen/blob/0.10.7/procgen/src/game.cpp#L56-L66
procgen_game_config = [
  ("bigfish", 6000, [0, 1]),
  ("bossfight", 4000, [0, 1]),
  ("caveflyer", 1000, [0, 1, 10]),
  ("chaser", 1000, [0, 1, 2]),
  ("climber", 1000, [0, 1]),
  ("coinrun", 1000, [0, 1]),
  ("dodgeball", 1000, [0, 1, 2, 10]),
  ("fruitbot", 1000, [0, 1]),
  ("heist", 1000, [0, 1, 10]),
  ("jumper", 1000, [0, 1, 10]),
  ("leaper", 500, [0, 1, 2]),
  ("maze", 500, [0, 1, 10]),
  ("miner", 1000, [0, 1, 10]),
  ("ninja", 1000, [0, 1]),
  ("plunder", 4000, [0, 1]),
  ("starpilot", 1000, [0, 1, 2]),
]

distribution = {
  0: "Easy",
  1: "Hard",
  2: "Extreme",
  10: "Memory",
}

for env_name, timeout, dist_mode in procgen_game_config:
  for dist_value in dist_mode:
    dist_name = distribution[dist_value]
    register(
      task_id=f"{env_name.capitalize()}{dist_name}-v0",
      import_path="envpool.procgen",
      spec_cls="ProcgenEnvSpec",
      dm_cls="ProcgenDMEnvPool",
      gym_cls="ProcgenGymEnvPool",
      gymnasium_cls="ProcgenGymnasiumEnvPool",
      env_name=env_name,
      distribution_mode=dist_value,
      max_episode_steps=timeout,
    )
