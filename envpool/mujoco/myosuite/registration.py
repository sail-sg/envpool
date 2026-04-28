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
"""Public MyoSuite task registration."""

from envpool.mujoco.myosuite.config import (
    myosuite_expanded_entry,
    resolve_myosuite_task_config,
)
from envpool.mujoco.myosuite.metadata import MYOSUITE_EXPANDED_IDS
from envpool.registration import register, registry

_IMPORT_PATH = "envpool.mujoco.myosuite.native"
MYOSUITE_PUBLIC_TASK_IDS = tuple(MYOSUITE_EXPANDED_IDS)

_CLASS_PREFIX = {
    ("myobase", "PoseEnvV0"): "MyoSuitePose",
    ("myobase", "ReachEnvV0"): "MyoSuiteReach",
    ("myobase", "Geometries100EnvV0"): "MyoSuiteReorient",
    ("myobase", "Geometries8EnvV0"): "MyoSuiteReorient",
    ("myobase", "InDistribution"): "MyoSuiteReorient",
    ("myobase", "OutofDistribution"): "MyoSuiteReorient",
    ("myobase", "KeyTurnEnvV0"): "MyoSuiteKeyTurn",
    ("myobase", "ObjHoldFixedEnvV0"): "MyoSuiteObjHold",
    ("myobase", "ObjHoldRandomEnvV0"): "MyoSuiteObjHold",
    ("myobase", "TorsoEnvV0"): "MyoSuiteTorso",
    ("myobase", "PenTwirlFixedEnvV0"): "MyoSuitePenTwirl",
    ("myobase", "PenTwirlRandomEnvV0"): "MyoSuitePenTwirl",
    ("myobase", "WalkEnvV0"): "MyoSuiteWalk",
    ("myobase", "TerrainEnvV0"): "MyoSuiteTerrain",
    ("myochallenge", "ReorientEnvV0"): "MyoChallengeReorient",
    ("myochallenge", "RelocateEnvV0"): "MyoChallengeRelocate",
    ("myochallenge", "BaodingEnvV1"): "MyoChallengeBaoding",
    ("myochallenge", "BimanualEnvV1"): "MyoChallengeBimanual",
    ("myochallenge", "RunTrack"): "MyoChallengeRunTrack",
    ("myochallenge", "SoccerEnvV0"): "MyoChallengeSoccer",
    ("myochallenge", "ChaseTagEnvV0"): "MyoChallengeChaseTag",
    ("myochallenge", "TableTennisEnvV0"): "MyoChallengeTableTennis",
    ("myodm", "TrackEnv"): "MyoDMTrack",
}


def _public_env_names(task_id: str) -> tuple[str, str, str]:
    entry, _ = myosuite_expanded_entry(task_id)
    prefix = _CLASS_PREFIX[(entry["suite"], entry["class_name"])]
    return (
        f"{prefix}EnvSpec",
        f"{prefix}DMEnvPool",
        f"{prefix}GymnasiumEnvPool",
    )


def register_myosuite_tasks() -> None:
    """Register every public MyoSuite task ID."""
    for task_id in MYOSUITE_PUBLIC_TASK_IDS:
        if task_id in registry.specs:
            continue
        entry, _ = myosuite_expanded_entry(task_id)
        spec_cls, dm_cls, gym_cls = _public_env_names(task_id)
        register(
            task_id=task_id,
            import_path=_IMPORT_PATH,
            spec_cls=spec_cls,
            dm_cls=dm_cls,
            gymnasium_cls=gym_cls,
            max_episode_steps=int(entry["max_episode_steps"]),
            _config_resolver=resolve_myosuite_task_config,
        )


register_myosuite_tasks()
