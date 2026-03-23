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
import sys
from typing import Any, Callable

import numpy as np
from jax import ShapeDtypeStruct, dtypes, ffi


def _normalize_specs(
    specs: tuple[tuple[Any, list[int]], ...],
) -> tuple[tuple[tuple[int, ...], Any], ...]:
    return tuple(
        (tuple(shape), dtypes.canonicalize_dtype(dtype))
        for dtype, shape in specs
    )


def _shape_dtype_struct(shape: tuple[int, ...], dtype: Any) -> ShapeDtypeStruct:
    return ShapeDtypeStruct(shape, dtype)


def _layout(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(range(len(shape)))


def _make_xla_function(
    obj: Any,
    handle: bytes,
    name: str,
    specs: tuple[tuple[Any, ...], tuple[Any, ...]],
    capsules: tuple[Any, Any],
) -> Callable:
    in_specs, out_specs = specs
    in_specs = _normalize_specs(in_specs)
    out_specs = _normalize_specs(out_specs)
    cpu_capsule, gpu_capsule = capsules
    call_target_name = f"{type(obj).__name__}_{id(obj)}_{name}"
    ffi.register_ffi_target(
        call_target_name,
        cpu_capsule,
        platform="cpu",
        api_version=1,
    )
    ffi.register_ffi_target(
        call_target_name,
        gpu_capsule,
        platform="gpu",
        api_version=1,
    )
    result_specs = tuple(_shape_dtype_struct(*spec) for spec in out_specs)
    xla_func = ffi.ffi_call(
        call_target_name,
        result_specs if len(result_specs) > 1 else result_specs[0],
        has_side_effect=True,
        input_layouts=tuple(_layout(shape) for shape, _ in in_specs),
        output_layouts=(
            tuple(_layout(shape) for shape, _ in out_specs)
            if len(out_specs) > 1
            else _layout(out_specs[0][0])
        ),
        input_output_aliases={0: 0},
    )
    handle_value = int.from_bytes(handle, byteorder=sys.byteorder, signed=False)

    def call(*args: Any) -> Any:
        return xla_func(*args, handle=handle_value)

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
        "XlaFunctions", ["handle", *method_names]
    )
    return XlaFunctions(  # type: ignore
        np.frombuffer(handle, dtype=np.uint8), *methods
    )
