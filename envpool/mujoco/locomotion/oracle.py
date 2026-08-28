# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pinned Composer factories used only by tests and documentation tooling."""

from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from dm_control.locomotion import soccer
from dm_control.locomotion.examples import (
    basic_cmu_2019,
    basic_rodent_2020,
    cmu_2020_tracking,
)
from dm_control.locomotion.mocap import cmu_mocap_data

EXAMPLE_MODULES = (basic_cmu_2019, basic_rodent_2020, cmu_2020_tracking)
_ORACLE_ASSETS = (
    Path(__file__).absolute().parents[3] / "third_party/dmc_locomotion"
)


def make_oracle(task: str, seed: int = 0, **kwargs: Any) -> Any:
    """Construct the pinned official factory with only its assets redirected."""
    # Upstream's maze/two-touch factories additionally use NumPy's module RNG.
    # EnvPool isolates that stream per environment, with the same seed.
    np.random.seed(seed)

    def mocap_path(version: str = "2020") -> str:
        return str(_ORACLE_ASSETS / f"oracle_{version}.h5")

    with mock.patch.object(cmu_mocap_data, "get_path_for_cmu", mocap_path):
        if task.startswith("soccer_"):
            return soccer.load(
                team_size=kwargs.pop("team_size", 2),
                walker_type=soccer.WalkerType[
                    task.removeprefix("soccer_").upper()
                ],
                random_state=seed,
                **kwargs,
            )
        module = next(
            module for module in EXAMPLE_MODULES if hasattr(module, task)
        )
        return getattr(module, task)(random_state=seed)


def oracle_observations(value: Any) -> dict[str, np.ndarray]:
    """Add the public player batch axis to official observation dictionaries."""
    if isinstance(value, list):
        return {
            key: np.stack([player[key] for player in value]) for key in value[0]
        }
    return {key: np.expand_dims(item, 0) for key, item in value.items()}
