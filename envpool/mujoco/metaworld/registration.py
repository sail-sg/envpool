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
"""MetaWorld v3 Sawyer env registration."""

from envpool.registration import register

metaworld_v3_envs = [
    "assembly-v3",
    "basketball-v3",
    "bin-picking-v3",
    "box-close-v3",
    "button-press-topdown-v3",
    "button-press-topdown-wall-v3",
    "button-press-v3",
    "button-press-wall-v3",
    "coffee-button-v3",
    "coffee-pull-v3",
    "coffee-push-v3",
    "dial-turn-v3",
    "disassemble-v3",
    "door-close-v3",
    "door-lock-v3",
    "door-open-v3",
    "door-unlock-v3",
    "hand-insert-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "faucet-open-v3",
    "faucet-close-v3",
    "hammer-v3",
    "handle-press-side-v3",
    "handle-press-v3",
    "handle-pull-side-v3",
    "handle-pull-v3",
    "lever-pull-v3",
    "pick-place-wall-v3",
    "pick-out-of-hole-v3",
    "pick-place-v3",
    "plate-slide-v3",
    "plate-slide-side-v3",
    "plate-slide-back-v3",
    "plate-slide-back-side-v3",
    "peg-insert-side-v3",
    "peg-unplug-side-v3",
    "soccer-v3",
    "stick-push-v3",
    "stick-pull-v3",
    "push-v3",
    "push-wall-v3",
    "push-back-v3",
    "reach-v3",
    "reach-wall-v3",
    "shelf-place-v3",
    "sweep-into-v3",
    "sweep-v3",
    "window-open-v3",
    "window-close-v3",
]

for task_name in metaworld_v3_envs:
    register(
        task_id=f"MetaWorld/{task_name}",
        import_path="envpool.mujoco.metaworld",
        spec_cls="MetaWorldEnvSpec",
        dm_cls="MetaWorldDMEnvPool",
        gymnasium_cls="MetaWorldGymnasiumEnvPool",
        aliases=[f"Meta-World/{task_name}"],
        task_name=task_name,
        max_episode_steps=500,
    )
