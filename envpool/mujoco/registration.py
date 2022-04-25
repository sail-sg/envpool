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

register(
  task_id="Ant-v4",
  import_path="envpool.mujoco",
  spec_cls="AntEnvSpec",
  dm_cls="AntDMEnvPool",
  gym_cls="AntGymEnvPool",
  max_episode_steps=1000,
  reward_threshold=6000.0,
  base_path=base_path,
)
