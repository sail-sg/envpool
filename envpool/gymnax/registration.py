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
"""Register Gymnax tasks with namespaced IDs and short aliases."""

from envpool.registration import package_base_path, register

from .gymnax_envpool import TASKS

for name, pool, limit in TASKS:
    register(
        task_id="Gymnax/" + name,
        aliases=(name,),
        import_path="envpool.gymnax",
        spec_cls=pool + "EnvSpec",
        dm_cls=pool + "DMEnvPool",
        gymnasium_cls=pool + "GymnasiumEnvPool",
        task=name,
        max_episode_steps=limit,
        base_path=package_base_path,
    )
