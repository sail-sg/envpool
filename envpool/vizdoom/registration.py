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
"""Vizdoom env registration."""

import os
from typing import List

from envpool.registration import base_path, register

maps_path = os.path.join(base_path, "vizdoom", "maps")


def _vizdoom_game_list() -> List[str]:
  return [
    game.replace(".cfg", "")
    for game in sorted(os.listdir(maps_path))
    if game.endswith(".cfg") and
    os.path.exists(os.path.join(maps_path, game.replace(".cfg", ".wad")))
  ]


for game in _vizdoom_game_list() + ["vizdoom_custom"]:
  name = "".join([g.capitalize() for g in game.split("_")])
  if game == "vizdoom_custom":
    cfg_path = wad_path = ""
  else:
    cfg_path = os.path.join(maps_path, f"{game}.cfg")
    wad_path = os.path.join(maps_path, f"{game}.wad")
  register(
    task_id=f"{name}-v1",
    import_path="envpool.vizdoom",
    spec_cls="VizdoomEnvSpec",
    dm_cls="VizdoomDMEnvPool",
    gym_cls="VizdoomGymEnvPool",
    gymnasium_cls="VizdoomGymnasiumEnvPool",
    cfg_path=cfg_path,
    wad_path=wad_path,
    max_episode_steps=525,
  )
