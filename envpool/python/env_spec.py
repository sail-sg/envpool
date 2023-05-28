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
"""EnvSpec mixin definition."""

import pprint
from abc import ABC, ABCMeta
from collections import namedtuple
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type, Union

import dm_env
import gym
import gymnasium

from .data import (
  dm_spec_transform,
  gym_spec_transform,
  gymnasium_spec_transform,
  to_namedtuple,
  to_nested_dict,
)
from .protocol import ArraySpec, EnvSpec
from .utils import check_key_duplication


class EnvSpecMixin(ABC):
  """Mixin class for EnvSpec, exposed to EnvSpecMeta."""

  gen_config: Type

  @property
  def config(self: EnvSpec) -> NamedTuple:
    """Configuration used to create the current EnvSpec."""
    return self.gen_config(*self._config_values)

  @property
  def reward_threshold(self: EnvSpec) -> Optional[float]:
    """Reward threshold, None for no threshold."""
    try:
      return self.config.reward_threshold  # type: ignore
    except AttributeError:
      return None

  @property
  def state_array_spec(self: EnvSpec) -> Dict[str, Any]:
    """Specs of the states of the environment.

    Returns:
      state_spec: A dict whose keys are the names of the states,
        its values is a tuple of (dtype, shape).
    """
    state_spec = [ArraySpec(*s) for s in self._state_spec]
    return dict(zip(self._state_keys, state_spec))

  @property
  def action_array_spec(self: EnvSpec) -> Dict[str, Any]:
    """Specs of the actions of the environment.

    Returns:
      state_spec: A dict whose keys are the names of the actions,
        its values is a tuple of (dtype, shape).
    """
    action_spec = [ArraySpec(*s) for s in self._action_spec]
    return dict(zip(self._action_keys, action_spec))

  def observation_spec(self: EnvSpec) -> Tuple:
    """Convert internal state_spec to dm_env compatible format.

    Returns:
      observation_spec: A namedtuple (maybe nested) that contains all keys
        that start with ``obs`` or ``info`` with their corresponding specs.
    """
    spec = self.state_array_spec
    spec = {
      k.replace("obs:", "").replace("info:", ""):
        dm_spec_transform(k.replace(":", ".").split(".")[-1], v, "obs")
      for k, v in spec.items()
      if k.startswith("obs") or k.startswith("info")
    }
    return to_namedtuple("State", to_nested_dict(spec))

  def action_spec(self: EnvSpec) -> Union[dm_env.specs.Array, Tuple]:
    """Convert internal action_spec to dm_env compatible format.

    Returns:
      action_spec: A single dm_env.specs.Array or a dict (maybe nested) that
        contains all keys that start with ``action`` with their corresponding
        specs.

    Note:
      If the original action_spec has a length of 3 ("env_id",
        "players.env_id", *), it returns the last spec instead of all for
        simplicity.
    """
    spec = self.action_array_spec
    if len(spec) == 3:
      # only env_id, players.env_id, action
      spec.pop("env_id")
      spec.pop("players.env_id")
      return dm_spec_transform(
        list(spec.keys())[0],
        list(spec.values())[0], "act"
      )
    spec = {
      k: dm_spec_transform(k.split(".")[-1], v, "act") for k, v in spec.items()
    }
    return to_namedtuple("Action", to_nested_dict(spec))

  @property
  def observation_space(self: EnvSpec) -> Union[gym.Space, Dict[str, Any]]:
    """Convert internal state_spec to gym.Env compatible format.

    Returns:
      observation_space: A dict (maybe nested) that contains all keys
        that start with ``obs`` with their corresponding specs.

    Note:
      If only one key starts with ``obs``, it returns that space instead of
        all for simplicity.
    """
    spec = self.state_array_spec
    spec = {
      k.replace("obs:", ""):
        gym_spec_transform(k.replace(":", ".").split(".")[-1], v, "obs")
      for k, v in spec.items()
      if k.startswith("obs")
    }
    if len(spec) == 1:
      return list(spec.values())[0]
    return to_nested_dict(spec, gym.spaces.Dict)

  @property
  def action_space(self: EnvSpec) -> Union[gym.Space, Dict[str, Any]]:
    """Convert internal action_spec to gym.Env compatible format.

    Returns:
      action_space: A dict (maybe nested) that contains key-value paired
        corresponding specs.

    Note:
      If the original action_spec has a length of 3 ("env_id",
        "players.env_id", *), it returns the last space instead of all for
        simplicity.
    """
    spec = self.action_array_spec
    if len(spec) == 3:
      # only env_id, players.env_id, action
      spec.pop("env_id")
      spec.pop("players.env_id")
      return gym_spec_transform(
        list(spec.keys())[0],
        list(spec.values())[0], "act"
      )
    spec = {
      k: gym_spec_transform(k.split(".")[-1], v, "act") for k, v in spec.items()
    }
    return to_nested_dict(spec, gym.spaces.Dict)

  @property
  def gymnasium_observation_space(
    self: EnvSpec
  ) -> Union[gymnasium.Space, Dict[str, Any]]:
    """Convert internal state_spec to gymnasium.Env compatible format.

    Returns:
      observation_space: A dict (maybe nested) that contains all keys
        that start with ``obs`` with their corresponding specs.

    Note:
      If only one key starts with ``obs``, it returns that space instead of
        all for simplicity.
    """
    spec = self.state_array_spec
    spec = {
      k.replace("obs:", ""):
        gymnasium_spec_transform(k.replace(":", ".").split(".")[-1], v, "obs")
      for k, v in spec.items()
      if k.startswith("obs")
    }
    if len(spec) == 1:
      return list(spec.values())[0]
    return to_nested_dict(spec, gymnasium.spaces.Dict)

  @property
  def gymnasium_action_space(
    self: EnvSpec
  ) -> Union[gymnasium.Space, Dict[str, Any]]:
    """Convert internal action_spec to gymnasium.Env compatible format.

    Returns:
      action_space: A dict (maybe nested) that contains key-value paired
        corresponding specs.

    Note:
      If the original action_spec has a length of 3 ("env_id",
        "players.env_id", *), it returns the last space instead of all for
        simplicity.
    """
    spec = self.action_array_spec
    if len(spec) == 3:
      # only env_id, players.env_id, action
      spec.pop("env_id")
      spec.pop("players.env_id")
      return gymnasium_spec_transform(
        list(spec.keys())[0],
        list(spec.values())[0], "act"
      )
    spec = {
      k: gymnasium_spec_transform(k.split(".")[-1], v, "act")
      for k, v in spec.items()
    }
    return to_nested_dict(spec, gymnasium.spaces.Dict)

  def __repr__(self: EnvSpec) -> str:
    """Prettify debug info."""
    config_info = pprint.pformat(self.config)[6:]
    return f"{self.__class__.__name__}{config_info}"


class EnvSpecMeta(ABCMeta):
  """Additional checker and wrapper for EnvSpec."""

  def __new__(cls: Any, name: str, parents: Tuple, attrs: Dict) -> Any:
    """Check keys and initialize namedtuple config."""
    base = parents[0]
    parents = (base, EnvSpecMixin)
    config_keys = base._config_keys
    check_key_duplication(name, "config", config_keys)
    config_keys: List[str] = list(
      map(lambda s: s.replace(".", "_"), config_keys)
    )
    defaults: Tuple = base._default_config_values
    attrs["gen_config"] = namedtuple("Config", config_keys, defaults=defaults)
    return super().__new__(cls, name, parents, attrs)
