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
"""Classic control env registration."""

from envpool.registration import register

register(
  task_id="CartPole-v0",
  import_path="envpool.classic_control",
  spec_cls="CartPoleEnvSpec",
  dm_cls="CartPoleDMEnvPool",
  gym_cls="CartPoleGymEnvPool",
  gymnasium_cls="CartPoleGymnasiumEnvPool",
  max_episode_steps=200,
  reward_threshold=195.0,
)

register(
  task_id="CartPole-v1",
  import_path="envpool.classic_control",
  spec_cls="CartPoleEnvSpec",
  dm_cls="CartPoleDMEnvPool",
  gym_cls="CartPoleGymEnvPool",
  gymnasium_cls="CartPoleGymnasiumEnvPool",
  max_episode_steps=500,
  reward_threshold=475.0,
)

register(
  task_id="Pendulum-v0",
  import_path="envpool.classic_control",
  spec_cls="PendulumEnvSpec",
  dm_cls="PendulumDMEnvPool",
  gym_cls="PendulumGymEnvPool",
  gymnasium_cls="PendulumGymnasiumEnvPool",
  version=0,
  max_episode_steps=200,
)

register(
  task_id="Pendulum-v1",
  import_path="envpool.classic_control",
  spec_cls="PendulumEnvSpec",
  dm_cls="PendulumDMEnvPool",
  gym_cls="PendulumGymEnvPool",
  gymnasium_cls="PendulumGymnasiumEnvPool",
  version=1,
  max_episode_steps=200,
)

register(
  task_id="MountainCar-v0",
  import_path="envpool.classic_control",
  spec_cls="MountainCarEnvSpec",
  dm_cls="MountainCarDMEnvPool",
  gym_cls="MountainCarGymEnvPool",
  gymnasium_cls="MountainCarGymnasiumEnvPool",
  max_episode_steps=200,
)

register(
  task_id="MountainCarContinuous-v0",
  import_path="envpool.classic_control",
  spec_cls="MountainCarContinuousEnvSpec",
  dm_cls="MountainCarContinuousDMEnvPool",
  gym_cls="MountainCarContinuousGymEnvPool",
  gymnasium_cls="MountainCarContinuousGymnasiumEnvPool",
  max_episode_steps=999,
)

register(
  task_id="Acrobot-v1",
  import_path="envpool.classic_control",
  spec_cls="AcrobotEnvSpec",
  dm_cls="AcrobotDMEnvPool",
  gym_cls="AcrobotGymEnvPool",
  gymnasium_cls="AcrobotGymnasiumEnvPool",
  max_episode_steps=500,
)
