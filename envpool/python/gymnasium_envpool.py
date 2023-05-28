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
"""EnvPool meta class for gymnasium.Env API."""

from abc import ABC, ABCMeta
from typing import Any, Dict, List, Tuple, Union

import gymnasium
import numpy as np
import optree

from .data import gymnasium_structure
from .envpool import EnvPoolMixin
from .utils import check_key_duplication


class GymnasiumEnvPoolMixin(ABC):
  """Special treatment for gymnasim API."""

  @property
  def observation_space(self: Any) -> Union[gymnasium.Space, Dict[str, Any]]:
    """Observation space from EnvSpec."""
    if not hasattr(self, "_gym_observation_space"):
      self._gym_observation_space = self.spec.gymnasium_observation_space
    return self._gym_observation_space

  @property
  def action_space(self: Any) -> Union[gymnasium.Space, Dict[str, Any]]:
    """Action space from EnvSpec."""
    if not hasattr(self, "_gym_action_space"):
      self._gym_action_space = self.spec.gymnasium_action_space
    return self._gym_action_space


class GymnasiumEnvPoolMeta(ABCMeta, gymnasium.Env.__class__):
  """Additional wrapper for EnvPool gymnasium.Env API."""

  def __new__(cls: Any, name: str, parents: Tuple, attrs: Dict) -> Any:
    """Check internal config and initialize data format convertion."""
    base = parents[0]
    try:
      from .lax import XlaMixin

      parents = (
        base, GymnasiumEnvPoolMixin, EnvPoolMixin, XlaMixin, gymnasium.Env
      )
    except ImportError:

      def _xla(self: Any) -> None:
        raise RuntimeError("XLA is disabled. To enable XLA please install jax.")

      attrs["xla"] = _xla
      parents = (base, GymnasiumEnvPoolMixin, EnvPoolMixin, gymnasium.Env)

    state_keys = base._state_keys
    action_keys = base._action_keys
    check_key_duplication(name, "state", state_keys)
    check_key_duplication(name, "action", action_keys)

    state_paths, state_idx, treepsec = gymnasium_structure(state_keys)

    def _to_gymnasium(
      self: Any, state_values: List[np.ndarray], reset: bool, return_info: bool
    ) -> Union[
      Any,
      Tuple[Any, Any],
      Tuple[Any, np.ndarray, np.ndarray, Any],
      Tuple[Any, np.ndarray, np.ndarray, np.ndarray, Any],
    ]:
      values = (state_values[i] for i in state_idx)
      state = optree.tree_unflatten(treepsec, values)
      info = state["info"]
      info["elapsed_step"] = state["elapsed_step"]
      if reset:
        return state["obs"], info
      terminated = state["done"] & ~state["trunc"]
      return state["obs"], state["reward"], terminated, state["trunc"], info

    attrs["_to"] = _to_gymnasium
    subcls = super().__new__(cls, name, parents, attrs)

    def init(self: Any, spec: Any) -> None:
      """Set self.spec to EnvSpecMeta."""
      super(subcls, self).__init__(spec)
      self.spec = spec

    setattr(subcls, "__init__", init)  # noqa: B010
    return subcls
