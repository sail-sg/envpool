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
"""Box2D env registration."""

from envpool.registration import register

register(
  task_id="CarRacing-v1",
  import_path="envpool.box2d",
  spec_cls="CarRacingEnvSpec",
  dm_cls="CarRacingDMEnvPool",
  gym_cls="CarRacingGymEnvPool",
  max_episode_steps=1000,
  reward_threshold=900,
)
