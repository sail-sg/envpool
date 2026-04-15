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
"""Generated MyoSuite surface metadata used by tests and registration."""

from __future__ import annotations

import json
from functools import cache
from typing import Any

from envpool.mujoco.myosuite.paths import myosuite_metadata_path


@cache
def load_myosuite_metadata() -> dict[str, Any]:
    """Load the generated MyoSuite metadata bundle."""
    return json.loads(myosuite_metadata_path().read_text())


_METADATA = load_myosuite_metadata()

MYOSUITE_PINS: dict[str, Any] = dict(_METADATA["pins"])
MYOSUITE_COUNTS: dict[str, int] = dict(_METADATA["counts"])
MYOSUITE_NOTES: dict[str, Any] = dict(_METADATA["notes"])
MYOSUITE_DIRECT_ENTRIES: tuple[dict[str, Any], ...] = tuple(
    _METADATA["direct_entries"]
)
MYOSUITE_DIRECT_ENTRY_BY_ID: dict[str, dict[str, Any]] = {
    entry["id"]: entry for entry in MYOSUITE_DIRECT_ENTRIES
}
MYOSUITE_DIRECT_IDS: tuple[str, ...] = tuple(_METADATA["direct_ids"])
MYOSUITE_EXPANDED_IDS: tuple[str, ...] = tuple(_METADATA["expanded_ids"])
MYOSUITE_SUITES: dict[str, Any] = dict(_METADATA["suites"])
