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
"""MyoSuite v2.11.6 env registration."""

from envpool.registration import asset_base_path, register

from .tasks import MYOSUITE_TASKS

myosuite_task_ids = [str(task["id"]) for task in MYOSUITE_TASKS]
myosuite_envpool_task_ids = [
    f"MyoSuite/{task_id}" for task_id in myosuite_task_ids
]
_MYOSUITE_BASE_PATH = asset_base_path(
    "envpool_assets_mujoco_large", "mujoco/myosuite/assets"
)

for task in MYOSUITE_TASKS:
    task_id = str(task["id"])
    register(
        task_id=task_id,
        aliases=(f"MyoSuite/{task_id}",),
        import_path="envpool.mujoco.myosuite",
        spec_cls="MyoSuiteEnvSpec",
        dm_cls="MyoSuiteDMEnvPool",
        gymnasium_cls="MyoSuiteGymnasiumEnvPool",
        task_name=task_id,
        max_episode_steps=task["max_episode_steps"],
        base_path=_MYOSUITE_BASE_PATH,
    )
