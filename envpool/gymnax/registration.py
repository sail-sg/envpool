# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Register Gymnax's native tasks and its variants of existing tasks."""

from typing import Any

from envpool.registration import package_base_path, register

from .gymnax_envpool import TASKS

for name, module, pool, limit in TASKS:
    options: dict[str, Any] = {}
    if module == "gymnax":
        options["task"] = name
        options["base_path"] = package_base_path
    elif pool == "Pendulum":
        options["version"] = 1
    elif pool == "MountainCarContinuous":
        options["stop_at_left_wall"] = False
    register(
        task_id="Gymnax/" + name,
        aliases=() if module == "classic_control" else (name,),
        import_path=f"envpool.{module}",
        spec_cls=pool + "EnvSpec",
        dm_cls=pool + "DMEnvPool",
        gymnasium_cls=pool + "GymnasiumEnvPool",
        max_episode_steps=limit,
        **options,
    )
