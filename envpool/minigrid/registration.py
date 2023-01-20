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
"""Minigrid env registration."""

from envpool.registration import register

register(
  task_id="MiniGrid-Empty-5x5-v0",
  import_path="envpool.minigrid",
  spec_cls="EmptyEnvSpec",
  dm_cls="EmptyDMEnvPool",
  gym_cls="EmptyGymEnvPool",
  gymnasium_cls="EmptyGymnasiumEnvPool",
  max_episode_steps=100,
  size=5,
)
