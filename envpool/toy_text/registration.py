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
  task_id="Catch-v0",
  import_path="envpool.toy_text",
  spec_cls="CatchEnvSpec",
  dm_cls="CatchDMEnvPool",
  gym_cls="CatchGymEnvPool",
  gymnasium_cls="CatchGymnasiumEnvPool",
  height=10,
  width=5,
)

register(
  task_id="FrozenLake-v1",
  import_path="envpool.toy_text",
  spec_cls="FrozenLakeEnvSpec",
  dm_cls="FrozenLakeDMEnvPool",
  gym_cls="FrozenLakeGymEnvPool",
  gymnasium_cls="FrozenLakeGymnasiumEnvPool",
  size=4,
  max_episode_steps=100,
  reward_threshold=0.7,
)

register(
  task_id="FrozenLake8x8-v1",
  import_path="envpool.toy_text",
  spec_cls="FrozenLakeEnvSpec",
  dm_cls="FrozenLakeDMEnvPool",
  gym_cls="FrozenLakeGymEnvPool",
  gymnasium_cls="FrozenLakeGymnasiumEnvPool",
  size=8,
  max_episode_steps=200,
  reward_threshold=0.85,
)

register(
  task_id="Taxi-v3",
  import_path="envpool.toy_text",
  spec_cls="TaxiEnvSpec",
  dm_cls="TaxiDMEnvPool",
  gym_cls="TaxiGymEnvPool",
  gymnasium_cls="TaxiGymnasiumEnvPool",
  max_episode_steps=200,
  reward_threshold=8.0,
)

register(
  task_id="NChain-v0",
  import_path="envpool.toy_text",
  spec_cls="NChainEnvSpec",
  dm_cls="NChainDMEnvPool",
  gym_cls="NChainGymEnvPool",
  gymnasium_cls="NChainGymnasiumEnvPool",
  max_episode_steps=1000,
)

register(
  task_id="CliffWalking-v0",
  import_path="envpool.toy_text",
  spec_cls="CliffWalkingEnvSpec",
  dm_cls="CliffWalkingDMEnvPool",
  gym_cls="CliffWalkingGymEnvPool",
  gymnasium_cls="CliffWalkingGymnasiumEnvPool",
)

register(
  task_id="Blackjack-v1",
  import_path="envpool.toy_text",
  spec_cls="BlackjackEnvSpec",
  dm_cls="BlackjackDMEnvPool",
  gym_cls="BlackjackGymEnvPool",
  gymnasium_cls="BlackjackGymnasiumEnvPool",
  sab=True,
  natural=False,
)
