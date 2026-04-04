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
"""Helper function for data convertion."""

import keyword
import re
from collections import namedtuple
from typing import Any, cast

import dm_env
import gymnasium
import numpy as np
import optree
from optree import PyTreeSpec

from .protocol import ArraySpec

ACTION_THRESHOLD = 2**20


def _maybe_scalar_int(value: Any) -> int | None:
    arr = np.asarray(value)
    if arr.size != 1:
        return None
    scalar = arr.item()
    if not np.isfinite(scalar):
        return None
    integer = int(scalar)
    if not np.isclose(scalar, integer):
        return None
    return integer


def _maybe_discrete_range(
    spec: ArraySpec, spec_type: str
) -> tuple[int, int] | None:
    if np.prod(np.abs(spec.shape)) != 1:
        return None
    minimum = _maybe_scalar_int(spec.minimum)
    maximum = _maybe_scalar_int(spec.maximum)
    if minimum is None or maximum is None or maximum >= ACTION_THRESHOLD:
        return None
    if spec_type == "act":
        if not (spec.is_discrete or np.issubdtype(spec.dtype, np.integer)):
            return None
    elif not np.issubdtype(spec.dtype, np.integer):
        return None
    return minimum, maximum - minimum + 1


def to_nested_dict(
    flatten_dict: dict[str, Any], generator: type = dict
) -> dict[str, Any]:
    """Convert a flat dict to a hierarchical dict.

    The input dict's hierarchy is denoted by ``.``.

    Example:
      ::

        >>> to_nested_dict({"a.b": 2333, "a.c": 666})
        {"a": {"b": 2333, "c": 666}}

    Args:
      flatten_dict: a dict whose keys list contains ``.`` for hierarchical
        representation.
      generator: a type of mapping. Default to ``dict``.
    """
    ret: dict[str, Any] = generator()
    for k, v in flatten_dict.items():
        segments = k.split(".")
        ptr = ret
        for s in segments[:-1]:
            keys = ptr.spaces if isinstance(ptr, gymnasium.spaces.Dict) else ptr
            if s not in keys:
                ptr[s] = generator()
            ptr = ptr[s]
        ptr[segments[-1]] = v
    return ret


def to_namedtuple(name: str, hdict: dict) -> tuple:
    """Convert a hierarchical dict to namedtuple."""
    typename = re.sub(r"\W", "_", name)
    if not typename or typename[0].isdigit() or keyword.iskeyword(typename):
        typename = f"_{typename}"
    field_names = []
    used_field_names: dict[str, int] = {}
    for key in hdict.keys():
        field = re.sub(r"\W", "_", key)
        if not field or field[0].isdigit() or keyword.iskeyword(field):
            field = f"_{field}"
        if field in used_field_names:
            used_field_names[field] += 1
            field = f"{field}_{used_field_names[field]}"
        else:
            used_field_names[field] = 0
        field_names.append(field)
    return namedtuple(typename, field_names)(*[
        to_namedtuple(k, v) if isinstance(v, dict) else v
        for k, v in hdict.items()
    ])


def dm_spec_transform(
    name: str, spec: ArraySpec, spec_type: str
) -> dm_env.specs.Array:
    """Transform ArraySpec to dm_env compatible specs."""
    discrete_range = _maybe_discrete_range(spec, spec_type)
    if discrete_range is not None and discrete_range[0] == 0:
        # dm_env only supports zero-based discrete arrays.
        return dm_env.specs.DiscreteArray(
            name=name,
            dtype=spec.dtype
            if np.issubdtype(spec.dtype, np.integer)
            else np.int32,
            num_values=discrete_range[1],
        )
    return dm_env.specs.BoundedArray(
        name=name,
        shape=[s for s in spec.shape if s != -1],
        dtype=spec.dtype,
        minimum=spec.minimum,
        maximum=spec.maximum,
    )


def gym_spec_transform(
    name: str, spec: ArraySpec, spec_type: str
) -> gymnasium.Space:
    """Transform ArraySpec to gym.Env compatible spaces."""
    discrete_range = _maybe_discrete_range(spec, spec_type)
    if discrete_range is not None:
        start, num_values = discrete_range
        return gymnasium.spaces.Discrete(n=num_values, start=start)
    return gymnasium.spaces.Box(
        shape=[s for s in spec.shape if s != -1],
        dtype=spec.dtype,
        low=spec.minimum,
        high=spec.maximum,
    )


def gymnasium_spec_transform(
    name: str, spec: ArraySpec, spec_type: str
) -> gymnasium.Space:
    """Transform ArraySpec to gymnasium.Env compatible spaces."""
    return gym_spec_transform(name, spec, spec_type)


def dm_structure(
    root_name: str,
    keys: list[str],
) -> tuple[list[tuple[int, ...]], list[int], PyTreeSpec]:
    """Convert flat keys into tree structure for namedtuple construction."""
    new_keys = []
    for key in keys:
        if key in ["obs", "info"]:  # special treatment for single-node obs/info
            key = f"obs:{key}"
        key = key.replace("info:", "obs:")  # merge obs and info together
        key = key.replace(
            "obs:", f"{root_name}:"
        )  # compatible with to_namedtuple
        new_keys.append(key.replace(":", "."))
    dict_tree = to_nested_dict(
        dict(zip(new_keys, list(range(len(new_keys))), strict=False))
    )
    structure = to_namedtuple(root_name, dict_tree)
    paths: list[tuple[int, ...]]
    indices: list[int]
    paths, indices, treespec = optree.tree_flatten_with_path(
        cast(Any, structure)
    )
    return paths, indices, treespec


def gym_structure(
    keys: list[str],
) -> tuple[list[tuple[str, ...]], list[int], PyTreeSpec]:
    """Convert flat keys into tree structure for dict construction."""
    keys = [k.replace(":", ".") for k in keys]
    dict_tree = to_nested_dict(
        dict(zip(keys, list(range(len(keys))), strict=False))
    )
    paths: list[tuple[str, ...]]
    indices: list[int]
    paths, indices, treespec = optree.tree_flatten_with_path(
        cast(Any, dict_tree)
    )
    return paths, indices, treespec


gymnasium_structure = gym_structure
