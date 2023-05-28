# Copyright 2021 Garena Online Private Limited
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
"""EnvPool meta class for dm_env API."""

from abc import ABC, ABCMeta
from typing import Any, Dict, List, Tuple, Union

import dm_env
import numpy as np
import optree
from dm_env import TimeStep

from .data import dm_structure
from .envpool import EnvPoolMixin
from .utils import check_key_duplication


class DMEnvPoolMixin(ABC):
  """Special treatment for dm_env API."""

  def observation_spec(self: Any) -> Tuple:
    """Observation spec from EnvSpec."""
    if not hasattr(self, "_dm_observation_spec"):
      self._dm_observation_spec = self.spec.observation_spec()
    return self._dm_observation_spec

  def action_spec(self: Any) -> Union[dm_env.specs.Array, Tuple]:
    """Action spec from EnvSpec."""
    if not hasattr(self, "_dm_action_spec"):
      self._dm_action_spec = self.spec.action_spec()
    return self._dm_action_spec


class DMEnvPoolMeta(ABCMeta):
  """Additional wrapper for EnvPool dm_env API."""

  def __new__(cls: Any, name: str, parents: Tuple, attrs: Dict) -> Any:
    """Check internal config and initialize data format convertion."""
    base = parents[0]
    try:
      from .lax import XlaMixin

      parents = (
        base, DMEnvPoolMixin, EnvPoolMixin, XlaMixin, dm_env.Environment
      )
    except ImportError:

      def _xla(self: Any) -> None:
        raise RuntimeError("XLA is disabled. To enable XLA please install jax.")

      attrs["xla"] = _xla
      parents = (base, DMEnvPoolMixin, EnvPoolMixin, dm_env.Environment)

    state_keys = base._state_keys
    action_keys = base._action_keys
    check_key_duplication(name, "state", state_keys)
    check_key_duplication(name, "action", action_keys)

    state_paths, state_idx, treepsec = dm_structure("State", state_keys)

    def _to_dm(
      self: Any,
      state_values: List[np.ndarray],
      reset: bool,
      return_info: bool,
    ) -> TimeStep:
      values = (state_values[i] for i in state_idx)
      state = optree.tree_unflatten(treepsec, values)
      timestep = TimeStep(
        step_type=state.step_type,
        observation=state.State,
        reward=state.reward,
        discount=state.discount,
      )
      return timestep

    attrs["_to"] = _to_dm
    subcls = super().__new__(cls, name, parents, attrs)

    def init(self: Any, spec: Any) -> None:
      """Set self.spec to EnvSpecMeta."""
      super(subcls, self).__init__(spec)
      self.spec = spec

    setattr(subcls, "__init__", init)  # noqa: B010
    return subcls
