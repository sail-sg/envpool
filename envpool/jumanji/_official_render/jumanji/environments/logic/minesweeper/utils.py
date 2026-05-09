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
"""NumPy implementation of the Minesweeper helper used by the viewer."""

from __future__ import annotations

from typing import Any

import numpy as np

from jumanji.environments.logic.minesweeper.constants import IS_MINE


def explored_mine(state: Any, action: np.ndarray) -> bool:
    row, col = np.asarray(action, dtype=np.int64)
    index = int(col + row * state.board.shape[-1])
    locations = np.asarray(state.flat_mine_locations, dtype=np.int64).reshape(-1)
    return bool(index in set(locations.tolist()))
