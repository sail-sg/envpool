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
"""Protocol of C++ EnvPool."""

from typing import (
  Any,
  Callable,
  Dict,
  List,
  NamedTuple,
  Optional,
  Tuple,
  Type,
  Union,
)

import dm_env
import gym
import numpy as np
from dm_env import TimeStep

try:
  from typing import Protocol
except ImportError:
  from typing_extensions import Protocol  # type: ignore


class EnvSpec(Protocol):
  """Cpp EnvSpec class."""

  _config_keys: List[str]
  _default_config_values: Tuple
  gen_config: Type

  def __init__(self, config: Tuple):
    """Protocol for constructor of EnvSpec."""

  @property
  def _state_spec(self) -> Tuple:
    """Cpp private _state_spec."""

  @property
  def _action_spec(self) -> Tuple:
    """Cpp private _action_spec."""

  @property
  def _state_keys(self) -> List:
    """Cpp private _state_keys."""

  @property
  def _action_keys(self) -> List:
    """Cpp private _action_keys."""

  @property
  def _config_values(self) -> Tuple:
    """Cpp private _config_values."""

  @property
  def config(self) -> NamedTuple:
    """Configuration used to create the current EnvSpec."""

  @property
  def state_array_spec(self) -> Dict[str, Any]:
    """Specs of the states of the environment in ArraySpec format."""

  @property
  def action_array_spec(self) -> Dict[str, Any]:
    """Specs of the actions of the environment in ArraySpec format."""

  def observation_spec(self) -> Dict[str, Any]:
    """Specs of the observations of the environment in dm_env format."""

  def action_spec(self) -> Union[dm_env.specs.Array, Dict[str, Any]]:
    """Specs of the actions of the environment in dm_env format."""

  @property
  def observation_space(self) -> Dict[str, Any]:
    """Specs of the observations of the environment in gym.Env format."""

  @property
  def action_space(self) -> Union[gym.Space, Dict[str, Any]]:
    """Specs of the actions of the environment in gym.Env format."""

  @property
  def reward_threshold(self) -> Optional[float]:
    """Reward threshold, None for no threshold."""


class ArraySpec(object):
  """Spec of numpy array."""

  def __init__(
    self, dtype: Type, shape: List[int], bounds: Tuple[Any, Any],
    element_wise_bounds: Tuple[Any, Any]
  ):
    """Constructor of ArraySpec."""
    self.dtype = dtype
    self.shape = shape
    if element_wise_bounds[0]:
      self.minimum = np.array(element_wise_bounds[0])
    else:
      self.minimum = bounds[0]
    if element_wise_bounds[1]:
      self.maximum = np.array(element_wise_bounds[1])
    else:
      self.maximum = bounds[1]

  def __repr__(self) -> str:
    """Beautify debug info."""
    return (
      f"ArraySpec(shape={self.shape}, dtype={self.dtype}, "
      f"minimum={self.minimum}, maximum={self.maximum})"
    )


class EnvPool(Protocol):
  """Cpp PyEnvpool class interface."""

  _state_keys: List[str]
  _action_keys: List[str]
  spec: Any

  def __init__(self, spec: EnvSpec):
    """Constructor of EnvPool."""

  def __len__(self) -> int:
    """Return the number of environments."""

  @property
  def _spec(self) -> EnvSpec:
    """Cpp env spec."""

  @property
  def _action_spec(self) -> List:
    """Cpp action spec."""

  def _check_action(self, actions: List) -> None:
    """Check action shapes."""

  def _recv(self) -> List[np.ndarray]:
    """Cpp private _recv method."""

  def _send(self, action: List[np.ndarray]) -> None:
    """Cpp private _send method."""

  def _reset(self, env_id: np.ndarray) -> None:
    """Cpp private _reset method."""

  def _from(
    self,
    action: Union[Dict[str, Any], np.ndarray],
    env_id: Optional[np.ndarray] = None,
  ) -> List[np.ndarray]:
    """Convertion for input action."""

  def _to(
    self,
    state: List[np.ndarray],
    reset: bool,
    return_info: bool,
  ) -> Union[TimeStep, Tuple]:
    """A switch of to_dm and to_gym for output state."""

  @property
  def all_env_ids(self) -> np.ndarray:
    """All env_id in numpy ndarray with dtype=np.int32."""

  @property
  def is_async(self) -> bool:
    """Return if this env is in sync mode or async mode."""

  @property
  def observation_space(self) -> Union[gym.Space, Dict[str, Any]]:
    """Gym observation space."""

  @property
  def action_space(self) -> Union[gym.Space, Dict[str, Any]]:
    """Gym action space."""

  def observation_spec(self) -> Tuple:
    """Dm observation spec."""

  def action_spec(self) -> Union[dm_env.specs.Array, Tuple]:
    """Dm action spec."""

  def seed(self, seed: Optional[Union[int, List[int]]] = None) -> None:
    """Set the seed for all environments."""

  @property
  def config(self) -> Dict[str, Any]:
    """Envpool config."""

  def send(
    self,
    action: Union[Dict[str, Any], np.ndarray],
    env_id: Optional[np.ndarray] = None,
  ) -> None:
    """Envpool send wrapper."""

  def recv(
    self,
    reset: bool = False,
    return_info: bool = True,
  ) -> Union[TimeStep, Tuple]:
    """Envpool recv wrapper."""

  def async_reset(self) -> None:
    """Envpool async reset interface."""

  def step(
    self,
    action: Union[Dict[str, Any], np.ndarray],
    env_id: Optional[np.ndarray] = None,
  ) -> Union[TimeStep, Tuple]:
    """Envpool step interface that performs send/recv."""

  def reset(
    self,
    env_id: Optional[np.ndarray] = None,
  ) -> Union[TimeStep, Tuple]:
    """Envpool reset interface."""

  def xla(self) -> Tuple[Any, Callable, Callable, Callable]:
    """Get the xla functions."""
