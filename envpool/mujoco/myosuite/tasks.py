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

"""MyoSuite task metadata generated from the pinned upstream source."""

from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path
from typing import Any, TypedDict, cast


class MyoSuiteTask(TypedDict):
    """Generated Python metadata for one pinned MyoSuite task."""

    id: str
    entry_point: str
    model_path: str
    reference_path: str
    object_name: str
    obs_dim: int
    action_dim: int
    max_episode_steps: int
    frame_skip: int
    normalize_act: bool
    oracle_numpy2_broken: bool


_METADATA_DIR = Path("assets/metadata")
_TASKS_JSON = "myosuite_tasks.json"
_ORACLE_JSON = "myosuite_oracle_metadata.json"


def _metadata_candidates(filename: str) -> tuple[Path, ...]:
    package_dir = Path(__file__).resolve().parent
    candidates = [package_dir / _METADATA_DIR / filename]
    runfiles = os.environ.get("TEST_SRCDIR")
    if runfiles:
        workspace = os.environ.get("TEST_WORKSPACE", "envpool")
        candidates.append(
            Path(runfiles)
            / workspace
            / "envpool/mujoco/myosuite"
            / _METADATA_DIR
            / filename
        )
    return tuple(dict.fromkeys(candidates))


def _read_metadata_json(filename: str) -> Any:
    attempted: list[str] = []
    for path in _metadata_candidates(filename):
        attempted.append(str(path))
        if path.is_file():
            return json.loads(path.read_text())
    package = __package__
    if package is None:
        raise FileNotFoundError(
            f"could not resolve package for MyoSuite metadata {filename}"
        )
    resource = (
        resources.files(package)
        .joinpath("assets")
        .joinpath("metadata")
        .joinpath(filename)
    )
    attempted.append(str(resource))
    if resource.is_file():
        return json.loads(resource.read_text())
    raise FileNotFoundError(
        f"could not find MyoSuite generated metadata {filename}; "
        f"tried {attempted}"
    )


MYOSUITE_TASKS = cast(list[MyoSuiteTask], _read_metadata_json(_TASKS_JSON))
_ORACLE_METADATA = cast(dict[str, object], _read_metadata_json(_ORACLE_JSON))
MYOSUITE_ORACLE_VERSION = str(_ORACLE_METADATA["version"])
MYOSUITE_ORACLE_COMMIT = str(_ORACLE_METADATA["commit"])
MYOSUITE_ORACLE_NUMPY2_BROKEN_IDS = frozenset(
    str(task_id)
    for task_id in cast(
        list[object], _ORACLE_METADATA["numpy2_broken_ids"]
    )
)

__all__ = [
    "MYOSUITE_ORACLE_COMMIT",
    "MYOSUITE_ORACLE_NUMPY2_BROKEN_IDS",
    "MYOSUITE_ORACLE_VERSION",
    "MYOSUITE_TASKS",
    "MyoSuiteTask",
]
