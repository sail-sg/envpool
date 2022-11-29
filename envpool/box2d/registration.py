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
"""Box2D env registration."""

from envpool.registration import register

register(
  task_id="CarRacing-v2",
  import_path="envpool.box2d",
  spec_cls="CarRacingEnvSpec",
  dm_cls="CarRacingDMEnvPool",
  gym_cls="CarRacingGymEnvPool",
  gymnasium_cls="CarRacingGymnasiumEnvPool",
  max_episode_steps=1000,
)

register(
  task_id="BipedalWalker-v3",
  import_path="envpool.box2d",
  spec_cls="BipedalWalkerEnvSpec",
  dm_cls="BipedalWalkerDMEnvPool",
  gym_cls="BipedalWalkerGymEnvPool",
  gymnasium_cls="BipedalWalkerGymnasiumEnvPool",
  hardcore=False,
  max_episode_steps=1600,
)

register(
  task_id="BipedalWalkerHardcore-v3",
  import_path="envpool.box2d",
  spec_cls="BipedalWalkerEnvSpec",
  dm_cls="BipedalWalkerDMEnvPool",
  gym_cls="BipedalWalkerGymEnvPool",
  gymnasium_cls="BipedalWalkerGymnasiumEnvPool",
  hardcore=True,
  max_episode_steps=2000,
)

register(
  task_id="LunarLander-v2",
  import_path="envpool.box2d",
  spec_cls="LunarLanderDiscreteEnvSpec",
  dm_cls="LunarLanderDiscreteDMEnvPool",
  gym_cls="LunarLanderDiscreteGymEnvPool",
  gymnasium_cls="LunarLanderDiscreteGymnasiumEnvPool",
  max_episode_steps=1000,
)

register(
  task_id="LunarLanderContinuous-v2",
  import_path="envpool.box2d",
  spec_cls="LunarLanderContinuousEnvSpec",
  dm_cls="LunarLanderContinuousDMEnvPool",
  gym_cls="LunarLanderContinuousGymEnvPool",
  gymnasium_cls="LunarLanderContinuousGymnasiumEnvPool",
  max_episode_steps=1000,
)
