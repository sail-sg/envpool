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
"""Verify every MyoSuite task's default seed/reset behavior against its oracle."""

from __future__ import annotations

import unittest
from typing import Any, cast

from envpool.mujoco.myosuite.myosuite_oracle_align_test import _run_oracle_probe
from envpool.mujoco.myosuite.myosuite_test import _SHARDED_TASKS
from envpool.python.seed_test_utils import check_seeded_resets


class MyoSuiteSeedTest(unittest.TestCase):
    """Check reproducibility and every random field observed in the oracle."""

    def test_default_reset_randomization_matches_oracle(self) -> None:
        """Check every task without synchronizing or replaying official state."""
        task_ids = tuple(task["id"] for task in _SHARDED_TASKS)
        report = _run_oracle_probe("reset_randomization", task_ids)
        tasks = cast(dict[str, dict[str, Any]], report["tasks"])
        for task_id in task_ids:
            with self.subTest(task_id=task_id):
                fields = {
                    key: cast(
                        tuple[bool | None, bool | None, bool | None],
                        tuple(True if flag else None for flag in flags),
                    )
                    for key, flags in tasks[task_id].items()
                }
                # Each randomized field must vary, not just any field in the
                # observation. Otherwise adding pose noise could hide a frozen goal.
                check_seeded_resets(
                    self,
                    task_id,
                    info_keys=tuple(key for key in fields if key != "obs"),
                    expected=cast(
                        tuple[bool | None, bool | None, bool | None],
                        tuple(
                            True
                            if any(flags[i] for flags in fields.values())
                            else None
                            for i in range(3)
                        ),
                    ),
                    field_expectations=fields,
                )


if __name__ == "__main__":
    # Tasks are already sharded by _SHARDED_TASKS; do not shard this single
    # test method a second time with absltest's automatic test loader.
    unittest.main()
