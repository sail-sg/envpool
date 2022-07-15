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
"""xla template on python side."""

from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Tuple, Union

import numpy as np
from jax import core, dtypes
from jax import numpy as jnp
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla
from jax.lib import xla_client


def _shape_with_layout(
  specs: Tuple[Tuple[List[int], Any], ...]
) -> Tuple[xla_client.Shape, ...]:
  return tuple(
    xla_client.Shape
    .array_shape(dtype, shape, tuple(range(len(shape) - 1, -1, -1)))
    if len(shape) > 0 else xla_client.Shape.scalar_shape(dtype)
    for shape, dtype in specs
  )


def _normalize_specs(
  specs: Tuple[Tuple[Any, List[int]], ...]
) -> Tuple[Tuple[List[int], Any], ...]:
  return tuple(
    (shape, dtypes.canonicalize_dtype(dtype)) for dtype, shape in specs
  )


def _make_xla_function(
  obj: Any, handle: bytes, name: str, specs: Tuple[Tuple[Any], Tuple[Any]],
  capsules: Tuple[Any, Any]
) -> Callable:
  in_specs, out_specs = specs
  in_specs = _normalize_specs(in_specs)
  out_specs = _normalize_specs(out_specs)
  cpu_capsule, gpu_capsule = capsules
  xla_client.register_cpu_custom_call_target(
    f"{type(obj).__name__}_{id(obj)}_{name}_cpu".encode(),
    cpu_capsule,
  )
  xla_client.register_custom_call_target(
    f"{type(obj).__name__}_{id(obj)}_{name}_gpu".encode(),
    gpu_capsule,
    platform="gpu",
  )

  def abstract(
    *args: List[jnp.ndarray]
  ) -> Union[ShapedArray, Tuple[ShapedArray, ...]]:
    if len(out_specs) > 1:
      return tuple(ShapedArray(*spec) for spec in out_specs)
    else:
      return ShapedArray(*out_specs[0])

  def translation(c: Any, *args: Any, platform: str = "cpu") -> Any:
    output_shape_with_layout = _shape_with_layout(out_specs)
    if len(out_specs) == 1:
      output_shape = output_shape_with_layout[0]
    else:
      output_shape = xla_client.Shape.tuple_shape(output_shape_with_layout)
    return xla_client.ops.CustomCallWithLayout(
      c,
      f"{type(obj).__name__}_{id(obj)}_{name}_{platform}".encode(),
      operands=args,
      operand_shapes_with_layout=_shape_with_layout(in_specs),
      shape_with_layout=output_shape,
      opaque=handle,
      has_side_effect=True,
    )

  prim = core.Primitive(f"{type(obj).__name__}_{id(obj)}_{name}")
  prim.multiple_results = (len(out_specs) > 1)
  prim.def_impl(partial(xla.apply_primitive, prim))
  prim.def_abstract_eval(abstract)
  xla.backend_specific_translations["cpu"][prim] = partial(
    translation, platform="cpu"
  )
  xla.backend_specific_translations["gpu"][prim] = partial(
    translation, platform="gpu"
  )

  def call(*args: Any) -> Any:
    return prim.bind(*args)

  return call


def make_xla(obj: Any) -> Any:
  """Return callables that can be jitted in a namedtuple.

  Args:
    obj: The object that has a `_xla` function.
      All instances of envpool has a `_xla` function that returns
      the necessary information for creating jittable send/recv functions.

  Returns:
    XlaFunctions: A namedtuple, the first element is a handle
      representing `obj`. The rest of the elements are jittable functions.
  """
  xla_native = obj._xla()
  method_names = []
  methods = []
  for name, (handle, specs, capsules) in xla_native:
    method_names.append(name)
    methods.append(_make_xla_function(obj, handle, name, specs, capsules))
  XlaFunctions = namedtuple(  # type: ignore
    "XlaFunctions",
    ["handle", *method_names]
  )
  return XlaFunctions(  # type: ignore
    np.frombuffer(handle, dtype=np.uint8),
    *methods
  )
