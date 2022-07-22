# Copyright 2022 Garena Online Private Limited
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
"""Provide xla mixin for envpool."""

from abc import ABC
from typing import Any, Callable, Dict, Optional, Tuple, Union

from dm_env import TimeStep
from jax import numpy as jnp

from .xla_template import make_xla


class XlaMixin(ABC):
  """Mixin to provide XLA for envpool class."""

  def xla(self: Any) -> Tuple[Any, Callable, Callable, Callable]:
    """Return the XLA version of send/recv/step functions."""
    _handle, _recv, _send = make_xla(self)

    def recv(handle: jnp.ndarray) -> Union[TimeStep, Tuple]:
      ret = _recv(handle)
      new_handle = ret[0]
      state_list = ret[1:]
      return new_handle, self._to(state_list, reset=False, return_info=True)

    def send(
      handle: jnp.ndarray,
      action: Union[Dict[str, Any], jnp.ndarray],
      env_id: Optional[jnp.ndarray] = None
    ) -> Any:
      action = self._from(action, env_id)
      self._check_action(action)
      return _send(handle, *action)

    def step(
      handle: jnp.ndarray,
      action: Union[Dict[str, Any], jnp.ndarray],
      env_id: Optional[jnp.ndarray] = None
    ) -> Any:
      return recv(send(handle, action, env_id))

    return _handle, recv, send, step
