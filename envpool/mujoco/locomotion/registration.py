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
"""Register every factory and Soccer walker from the pinned upstream source."""

from envpool.registration import asset_base_path, register

from .locomotion_envpool import TASKS

for task in TASKS:
    name = "".join(word.capitalize() for word in task.split("_"))
    register(
        task_id=f"Dmc{name}-v1",
        aliases=(f"dm_control/locomotion/{task}",),
        import_path="envpool.mujoco.locomotion",
        spec_cls="LocomotionEnvSpec",
        dm_cls="LocomotionDMEnvPool",
        gymnasium_cls="LocomotionGymnasiumEnvPool",
        task_name=task,
        base_path=asset_base_path(
            "envpool", "mujoco/locomotion/assets_dm_control"
        ),
    )
