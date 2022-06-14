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
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from dm_env import TimeStep
from jax import core, dtypes
from jax import numpy as jnp
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla
from jax.lib import xla_client

from .protocol import EnvPool


def has_dynamic_shape(spec: Tuple) -> bool:
  """Check whether the shape is dynamic."""
  shape = spec[1]
  if len(shape) == 0:  # scalar shape ()
    return False
  if isinstance(shape[0], list):
    # container shape ((1,2), (-1, 2))
    # the first element describes how the container is organized
    # the second describes the shape in each container
    return True
  else:
    # or, if any dimension other than the first has a shape -1
    return any(map(lambda v: v == -1, shape[1:]))


def normalize_shape(shape: List[int], batch_size: int,
                    max_num_players: int) -> List[int]:
  """Replace unknown first dim with batch_size/max_num_players."""
  if len(shape) > 0 and shape[0] == -1:
    return [batch_size * max_num_players, *shape[1:]]
  else:
    return [batch_size, *shape]


class XlaMixin(ABC):
  """Mixin to provide XLA for envpool class."""

  def _handle_spec(self: EnvPool) -> Tuple[List[int], Any]:
    return [2], dtypes.canonicalize_dtype(np.uint32)

  def _states_spec(self: EnvPool) -> List[Tuple[List[int], Any]]:
    batch_size = self.config["batch_size"]
    max_num_players = self.config["max_num_players"]
    specs = []
    for name, spec in zip(self._spec._state_keys, self._spec._state_spec):
      if has_dynamic_shape(spec):
        raise RuntimeError(
          f"XLA is disabled because state[{name}] has dynamic shape"
        )
      specs.append(
        (
          normalize_shape(spec[1], batch_size, max_num_players),
          dtypes.canonicalize_dtype(spec[0]),
        )
      )
    return specs

  def _actions_spec(self: EnvPool) -> List[Tuple[List[int], Any]]:
    batch_size = self.config["batch_size"]
    max_num_players = self.config["max_num_players"]
    specs = []
    for name, spec in zip(self._spec._action_keys, self._spec._action_spec):
      if has_dynamic_shape(spec):
        raise RuntimeError(
          f"XLA is disabled because action[{name}] has dynamic shape"
        )
      specs.append(
        (
          normalize_shape(spec[1], batch_size, max_num_players),
          dtypes.canonicalize_dtype(spec[0]),
        )
      )
    return specs

  @staticmethod
  def _shape_with_layout(
    specs: List[Tuple[List[int], Any]]
  ) -> Tuple[xla_client.Shape, ...]:
    return tuple(
      xla_client.Shape
      .array_shape(dtype, shape, tuple(range(len(shape) - 1, -1, -1)))
      for shape, dtype in specs
    )

  def _send_abstract(
    self: Any,
    handle: jnp.ndarray,
    *action: List[jnp.ndarray],
  ) -> ShapedArray:
    return ShapedArray(*self._handle_spec())

  def _send_translation(
    self: Any,
    c: Any,
    handle: Any,
    *action: Any,
    platform: str = "cpu",
  ) -> Any:
    operand_specs = [self._handle_spec(), *self._actions_spec()]
    return xla_client.ops.CustomCallWithLayout(
      c,
      f"envpool_{id(self)}_{platform}_send".encode(),
      operands=(handle, *action),
      operand_shapes_with_layout=(self._shape_with_layout(operand_specs)),
      shape_with_layout=self._shape_with_layout([self._handle_spec()])[0],
      # otherwise the function might get called multiple times
      has_side_effect=True,
    )

  def _recv_abstract(self: Any, handle: Any) -> Any:
    return (ShapedArray(*self._handle_spec()),) + tuple(
      ShapedArray(*st) for st in self._states_spec()
    )

  def _recv_translation(
    self: Any,
    c: Any,
    handle: Any,
    *,
    platform: str = "cpu",
  ) -> Any:
    operand_shape = self._shape_with_layout([self._handle_spec()])
    output_specs = [self._handle_spec(), *self._states_spec()]
    output_shape = xla_client.Shape.tuple_shape(
      self._shape_with_layout(output_specs)
    )
    return xla_client.ops.CustomCallWithLayout(
      c,
      f"envpool_{id(self)}_{platform}_recv".encode(),
      operands=(handle,),
      operand_shapes_with_layout=operand_shape,
      shape_with_layout=output_shape,
      # otherwise the function might get called multiple times
      has_side_effect=True,
    )

  def _setup_send(self: Any) -> None:
    self._send_prim = _send_prim = core.Primitive(f"envpool_{id(self)}_send")
    _send_prim.multiple_results = False
    _send_prim.def_impl(partial(xla.apply_primitive, _send_prim))
    _send_prim.def_abstract_eval(self._send_abstract)
    xla.backend_specific_translations["cpu"][_send_prim] = partial(
      self._send_translation, platform="cpu"
    )

  def _setup_recv(self: Any) -> None:
    self._recv_prim = _recv_prim = core.Primitive(f"envpool_{id(self)}_recv")
    _recv_prim.multiple_results = True
    _recv_prim.def_impl(partial(xla.apply_primitive, _recv_prim))
    _recv_prim.def_abstract_eval(self._recv_abstract)
    xla.backend_specific_translations["cpu"][_recv_prim] = partial(
      self._recv_translation, platform="cpu"
    )

  def xla(self: Any) -> Tuple[Any, Callable, Callable, Callable]:
    """Return the XLA version of send/recv/step functions."""
    if not hasattr(self, "_xla_setup"):
      _handle, _send, _recv = self._xla()
      xla_client.register_cpu_custom_call_target(
        f"envpool_{id(self)}_cpu_recv", _recv
      )
      xla_client.register_cpu_custom_call_target(
        f"envpool_{id(self)}_cpu_send", _send
      )
      self._setup_recv()
      self._setup_send()
      self._xla_setup = True
      self._xla_handle = _handle

    def recv(handle: jnp.ndarray) -> Union[TimeStep, Tuple]:
      ret = self._recv_prim.bind(handle)
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
      return self._send_prim.bind(handle, *action)

    def step(
      handle: jnp.ndarray,
      action: Union[Dict[str, Any], jnp.ndarray],
      env_id: Optional[jnp.ndarray] = None
    ) -> Any:
      return recv(send(handle, action, env_id))

    return self._xla_handle, recv, send, step
