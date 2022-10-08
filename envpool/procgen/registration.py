# Copyright 2021 Garena Online Private Limited
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

from envpool.registration import register

# in total 16 games in Procgen
procgen_games_list = [
  "bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun",
  "dodgeball", "fruitbot", "heist", "jumper", "leaper", "maze", "miner",
  "ninja", "plunder", "starpilot"
]

distribution_name = ["easy", "hard"]

distribution_code = [0, 1]

for game_name in procgen_games_list:
  for dist_name in distribution_name:
    for dist_code in distribution_code:
      register(
        task_id=f"procgen-{game_name}-{dist_name}",
        import_path="envpool.procgen",
        spec_cls="ProcgenEnvSpec",
        dm_cls="ProcgenDMEnvPool",
        gym_cls="ProcgenGymEnvPool",
        game_name=game_name,
        num_levels=0,
        start_level=0,
        distribution_mode=dist_code
      )
