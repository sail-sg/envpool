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
"""Seed/reset behavior for every registered task, with aliases deduplicated."""

from __future__ import annotations

import json
import re
from contextlib import chdir
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from absl.testing import absltest, parameterized

import envpool  # noqa: F401
from envpool.python.seed_test_utils import (
    check_seeded_resets,
    check_seeded_rollouts,
)
from envpool.registration import registry


def _cases() -> list[tuple[str, str, str, dict[str, Any]]]:
    cases = []
    seen = set()
    for task_id, (module, spec, kwargs) in sorted(registry.specs.items()):
        # MyoSuite's fixed goals, randomized poses, and randomized model
        # parameters are checked against its pinned oracle in myosuite_seed_test.
        if module == "envpool.mujoco.myosuite":
            continue
        key = (module, spec, json.dumps(kwargs, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        name = re.sub(
            r"\W", "_", f"{module.removeprefix('envpool.')}_{task_id}"
        )
        cases.append((name, task_id, module, kwargs))
    # absltest shards these parameterized test cases using Bazel's protocol.
    return cases


def _behavior(task: str, module: str, config: dict[str, Any]) -> str:
    task = task.rsplit("/", 1)[-1]
    if module == "envpool.procgen" and config["distribution_mode"] == 20:
        # Upstream exploration mode deliberately fixes one level per game.
        return "fixed"
    if module == "envpool.minigrid" and (
        re.fullmatch(r"MiniGrid-Empty-\d+x\d+-v0", task)
        or task.startswith("MiniGrid-DistShift")
    ):
        # These upstream generators use fixed walls, agent pose, and goal.
        return "fixed"
    if module == "envpool.toy_text":
        if task in {"CliffWalking-v0", "CliffWalking-v1"}:
            return "fixed"
        if task.startswith(("FrozenLake", "NChain", "CliffWalkingSlippery")):
            return "rollout"
    if module in {"envpool.gfootball", "envpool.vizdoom"}:
        return "rollout"
    if module == "envpool.jumanji" and task in {"Minesweeper-v0", "PacMan-v1"}:
        # Mines are hidden on reset; the fixed PacMan maze has stochastic ghosts.
        return "rollout"
    return "reset"


def _info_keys(module: str, spec: Any) -> tuple[str, ...]:
    if module == "envpool.atari":
        # A ROM's title screen may not reveal randomized no-op advancement.
        return ("ram",)
    if (
        module == "envpool.pgx"
        and "info:current_player" in spec.state_array_spec
    ):
        # Empty boards can be identical while the randomly chosen first player differs.
        return ("current_player",)
    if module == "envpool.mujoco.metaworld":
        # Partially observable tasks hide the goal from the initial observation.
        return ("target0",)
    return ()


def _highway_state(pool: Any) -> Any:
    # Occupancy grids quantize vehicle positions. Inspect the physical state
    # too, so different positions in the same cells are not false failures.
    return np.array(
        [
            [(v.x, v.y, v.heading, v.speed) for v in state.vehicles]
            for state in pool.debug_states()
        ],
        dtype=object,
    )


class SeedTest(parameterized.TestCase):
    """Check simulated state, not only RNG/seed metadata or matching shapes."""

    @parameterized.named_parameters(_cases())
    def test_seed_behavior(
        self, task_id: str, module: str, config: dict[str, Any]
    ) -> None:
        """Exercise the task's reset or rollout randomness with controlled actions."""
        behavior = _behavior(task_id, module, config)
        kwargs: dict[str, Any] = {}
        if module == "envpool.vizdoom":
            # The engine writes _vizdoom in cwd; avoid collisions between Bazel
            # shards, as in the existing VizDoom render harness.
            directory = self.enterContext(
                TemporaryDirectory(prefix="envpool-seed-doom-")
            )
            self.enterContext(chdir(directory))
        if module == "envpool.vizdoom" and not config["cfg_path"]:
            # The custom factory requires a caller-supplied scenario.
            basic = registry.specs["Basic-v1"][2]
            kwargs.update(
                cfg_path=basic["cfg_path"], wad_path=basic["wad_path"]
            )
        spec = registry.make_spec(task_id, **kwargs)
        info_keys = _info_keys(module, spec)
        expected = {
            "reset": (True, True, True),
            "fixed": (False, False, False),
            "rollout": (None, None, None),
        }[behavior]
        check_seeded_resets(
            self,
            task_id,
            info_keys=info_keys,
            expected=expected,
            extra_state=_highway_state if module == "envpool.highway" else None,
            **kwargs,
        )
        if behavior != "reset":
            check_seeded_rollouts(
                self,
                task_id,
                info_keys=info_keys,
                expect_different=behavior != "fixed",
                **kwargs,
            )


if __name__ == "__main__":
    absltest.main()
