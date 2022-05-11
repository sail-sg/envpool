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
"""EnvPool meta class for gym.Env API."""

from abc import ABC, ABCMeta
from typing import Any, Dict, List, Tuple, Union, no_type_check

import gym
import numpy as np
import tree

from .data import gym_structure
from .envpool import EnvPoolMixin
from .utils import check_key_duplication


class GymEnvPoolMixin(ABC):
  """Special treatment for gym API."""

  @property
  def observation_space(self: Any) -> Union[gym.Space, Dict[str, Any]]:
    """Observation space from EnvSpec."""
    return self.spec.observation_space

  @property
  def action_space(self: Any) -> Union[gym.Space, Dict[str, Any]]:
    """Action space from EnvSpec."""
    return self.spec.action_space


class GymEnvPoolMeta(ABCMeta):
  """Additional wrapper for EnvPool gym.Env API."""

  def __new__(cls: Any, name: str, parents: Tuple, attrs: Dict) -> Any:
    """Check internal config and initialize data format convertion."""
    base = parents[0]
    parents = (base, GymEnvPoolMixin, EnvPoolMixin, gym.Env)
    state_keys = base._state_keys
    action_keys = base._action_keys
    check_key_duplication(name, "state", state_keys)
    check_key_duplication(name, "action", action_keys)

    state_structure, state_idx = gym_structure(state_keys)

    def _to_gym(
      self: Any, state_values: List[np.ndarray], reset: bool, return_info: bool
    ) -> Union[Any, Tuple[Any, Any], Tuple[Any, np.ndarray, np.ndarray, Any]]:
      state = tree.unflatten_as(
        state_structure, [state_values[i] for i in state_idx]
      )
      if reset and not return_info:
        return state["obs"]
      done = state["done"]
      elapse = state["elapsed_step"]
      max_episode_steps = self.config.get("max_episode_steps", np.inf)
      trunc = (done & (elapse >= max_episode_steps))
      state["info"]["TimeLimit.truncated"] = trunc
      if reset:
        return state["obs"], state["info"]
      return state["obs"], state["reward"], state["done"], state["info"]

    attrs["_to"] = _to_gym
    subcls = super().__new__(cls, name, parents, attrs)

    @no_type_check
    def init(self: Any, spec: Any) -> None:
      """Set self.spec to EnvSpecMeta."""
      super(subcls, self).__init__(spec)
      self.spec = spec

    setattr(subcls, "__init__", init)  # noqa: B010
    return subcls
