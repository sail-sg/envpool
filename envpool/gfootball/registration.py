# Copyright 2026 Garena Online Private Limited
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
"""Google Research Football env registration."""

from envpool.registration import register

_COMMON = {
    "import_path": "envpool.gfootball",
    "spec_cls": "GfootballEnvSpec",
    "dm_cls": "GfootballDMEnvPool",
    "gymnasium_cls": "GfootballGymnasiumEnvPool",
    "render": False,
}

_SCENARIOS = (
    ("11_vs_11_competition", 3000),
    ("11_vs_11_easy_stochastic", 3000),
    ("11_vs_11_hard_stochastic", 3000),
    ("11_vs_11_kaggle", 3000),
    ("11_vs_11_stochastic", 3000),
    ("1_vs_1_easy", 500),
    ("5_vs_5", 3000),
    ("academy_3_vs_1_with_keeper", 400),
    ("academy_corner", 400),
    ("academy_counterattack_easy", 400),
    ("academy_counterattack_hard", 400),
    ("academy_empty_goal", 400),
    ("academy_empty_goal_close", 400),
    ("academy_pass_and_shoot_with_keeper", 400),
    ("academy_run_pass_and_shoot_with_keeper", 400),
    ("academy_run_to_score", 400),
    ("academy_run_to_score_with_keeper", 400),
    ("academy_single_goal_versus_lazy", 3000),
)

for env_name, max_episode_steps in _SCENARIOS:
    register(
        task_id=f"gfootball/{env_name}-v1",
        env_name=env_name,
        max_episode_steps=max_episode_steps,
        **_COMMON,
    )
