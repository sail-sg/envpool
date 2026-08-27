# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Register every environment name accepted by the pinned official factory."""

from envpool.registration import register

from ._registry import CRAFTAX_IDS

CRAFTAX_ORACLE_VERSION = "1.6.1"
CRAFTAX_ORACLE_COMMIT = "c3c2e0d038c4e641f9481320c158f457f30c28f3"

for task_id in CRAFTAX_IDS:
    classic = "-Classic-" in task_id
    pixels = "-Pixels-" in task_id
    prefix = (
        "Craftax"
        + ("Classic" if classic else "")
        + ("Pixels" if pixels else "Symbolic")
    )
    register(
        task_id=task_id,
        aliases=("Craftax/" + task_id.removeprefix("Craftax-"),),
        import_path="envpool.craftax",
        spec_cls=prefix + "EnvSpec",
        dm_cls=prefix + "DMEnvPool",
        gymnasium_cls=prefix + "GymnasiumEnvPool",
        max_episode_steps=10000 if classic else 100000,
        auto_reset="-AutoReset-" in task_id,
    )
