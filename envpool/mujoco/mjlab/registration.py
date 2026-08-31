# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Register every built-in task from the pinned official MJLab registry."""

from envpool.registration import asset_base_path, register

from . import TASKS

for task in TASKS:
    register(
        task_id=f"{task}-v0",
        aliases=(task,),
        import_path="envpool.mujoco.mjlab",
        spec_cls="MjlabEnvSpec",
        dm_cls="MjlabDMEnvPool",
        gymnasium_cls="MjlabGymnasiumEnvPool",
        task_name=task,
        base_path=asset_base_path(
            "envpool_assets_mjlab", "mujoco/mjlab/assets"
        ),
    )
