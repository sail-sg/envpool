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
"""Runfile helpers for vendored MyoSuite assets and metadata."""

from __future__ import annotations

import os
from pathlib import Path

_WORKSPACE_NAME = "envpool"


def _candidate_roots() -> list[Path]:
    roots: list[Path] = [Path.cwd()]
    for env_name in ("RUNFILES_DIR", "TEST_SRCDIR"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))
    here = Path(__file__).resolve()
    roots.extend(here.parents)
    return roots


def resolve_workspace_path(relative_path: str) -> Path:
    """Resolve a workspace-relative path in repo or Bazel runfiles layouts."""
    rel = Path(relative_path)
    seen: set[Path] = set()
    for root in _candidate_roots():
        for candidate in (root / rel, root / _WORKSPACE_NAME / rel):
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"Unable to resolve workspace path {relative_path!r}; "
        f"checked {len(seen)} candidate locations."
    )


def myosuite_asset_root() -> Path:
    """Return the staged MyoSuite asset tree root."""
    return resolve_workspace_path("envpool/mujoco/myosuite_assets")


def myosuite_metadata_path() -> Path:
    """Return the vendored generated MyoSuite metadata JSON path."""
    try:
        return resolve_workspace_path(
            "third_party/myosuite/metadata/env_ids.json"
        )
    except FileNotFoundError as workspace_error:
        # Release wheels do not carry the source workspace, but Bazel packages
        # this generated metadata next to the Python modules.
        packaged_metadata = Path(__file__).resolve().with_name("env_ids.json")
        if packaged_metadata.exists():
            return packaged_metadata
        raise workspace_error
