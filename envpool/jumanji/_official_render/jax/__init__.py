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
"""Small JAX compatibility shim used only by vendored Jumanji viewers."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace
from typing import Any, Callable

import numpy as _np

from . import numpy


def device_get(value: Any) -> Any:
    return value


def jit(fn: Callable[..., Any] | None = None, **_: Any) -> Any:
    if fn is None:
        return lambda wrapped: wrapped
    return fn


def vmap(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Callable[..., Any]:
    del args, kwargs

    def mapped(first: Any, *rest: Any) -> Any:
        return _np.asarray([fn(item, *rest) for item in first])

    return mapped


def _tree_map(fn: Callable[..., Any], *trees: Any) -> Any:
    tree = trees[0]
    if is_dataclass(tree):
        values = {
            field.name: _tree_map(
                fn, *(getattr(item, field.name) for item in trees)
            )
            for field in fields(tree)
        }
        return replace(tree, **values)
    if isinstance(tree, tuple) and hasattr(tree, "_fields"):
        return type(tree)(
            *[
                _tree_map(fn, *(getattr(item, name) for item in trees))
                for name in tree._fields
            ]
        )
    return fn(*trees)


def _dynamic_slice_in_dim(
    value: Any, start_index: int, slice_size: int, axis: int = 0
) -> Any:
    slices = [slice(None)] * _np.asarray(value).ndim
    slices[axis] = slice(start_index, start_index + slice_size)
    return value[tuple(slices)]


def _switch(index: int, branches: list[Callable[..., Any]], *operands: Any) -> Any:
    return branches[int(index)](*operands)


tree_util = SimpleNamespace(tree_map=_tree_map)
lax = SimpleNamespace(
    dynamic_slice_in_dim=_dynamic_slice_in_dim,
    switch=_switch,
)
random = SimpleNamespace()
