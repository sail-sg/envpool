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

from collections import namedtuple
from typing import Any, Dict, List, Tuple, Type

import dm_env
import gym
import gymnasium
import numpy as np
import optree
from optree import PyTreeSpec

from .protocol import ArraySpec

ACTION_THRESHOLD = 2**20


def to_nested_dict(flatten_dict: Dict[str, Any],
                   generator: Type = dict) -> Dict[str, Any]:
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
  ret: Dict[str, Any] = generator()
  for k, v in flatten_dict.items():
    segments = k.split(".")
    ptr = ret
    for s in segments[:-1]:
      if s not in ptr:
        ptr[s] = generator()
      ptr = ptr[s]
    ptr[segments[-1]] = v
  return ret


def to_namedtuple(name: str, hdict: Dict) -> Tuple:
  """Convert a hierarchical dict to namedtuple."""
  return namedtuple(name, hdict.keys())(
    *[
      to_namedtuple(k, v) if isinstance(v, Dict) else v
      for k, v in hdict.items()
    ]
  )


def dm_spec_transform(
  name: str, spec: ArraySpec, spec_type: str
) -> dm_env.specs.Array:
  """Transform ArraySpec to dm_env compatible specs."""
  if np.prod(np.abs(spec.shape)) == 1 and \
      np.isclose(spec.minimum, 0) and spec.maximum < ACTION_THRESHOLD:
    # special treatment for discrete action space
    return dm_env.specs.DiscreteArray(
      name=name,
      dtype=spec.dtype,
      num_values=int(spec.maximum - spec.minimum + 1),
    )
  return dm_env.specs.BoundedArray(
    name=name,
    shape=[s for s in spec.shape if s != -1],
    dtype=spec.dtype,
    minimum=spec.minimum,
    maximum=spec.maximum,
  )


def gym_spec_transform(name: str, spec: ArraySpec, spec_type: str) -> gym.Space:
  """Transform ArraySpec to gym.Env compatible spaces."""
  if np.prod(np.abs(spec.shape)) == 1 and \
      np.isclose(spec.minimum, 0) and spec.maximum < ACTION_THRESHOLD:
    # special treatment for discrete action space
    discrete_range = int(spec.maximum - spec.minimum + 1)
    try:
      return gym.spaces.Discrete(n=discrete_range, start=int(spec.minimum))
    except TypeError:  # old gym version doesn't have `start`
      return gym.spaces.Discrete(n=discrete_range)
  return gym.spaces.Box(
    shape=[s for s in spec.shape if s != -1],
    dtype=spec.dtype,
    low=spec.minimum,
    high=spec.maximum,
  )


def gymnasium_spec_transform(
  name: str, spec: ArraySpec, spec_type: str
) -> gymnasium.Space:
  """Transform ArraySpec to gymnasium.Env compatible spaces."""
  if np.prod(np.abs(spec.shape)) == 1 and \
      np.isclose(spec.minimum, 0) and spec.maximum < ACTION_THRESHOLD:
    # special treatment for discrete action space
    discrete_range = int(spec.maximum - spec.minimum + 1)
    return gymnasium.spaces.Discrete(n=discrete_range, start=int(spec.minimum))
  return gymnasium.spaces.Box(
    shape=[s for s in spec.shape if s != -1],
    dtype=spec.dtype,
    low=spec.minimum,
    high=spec.maximum,
  )


def dm_structure(
  root_name: str,
  keys: List[str],
) -> Tuple[List[Tuple[int, ...]], List[int], PyTreeSpec]:
  """Convert flat keys into tree structure for namedtuple construction."""
  new_keys = []
  for key in keys:
    if key in ["obs", "info"]:  # special treatment for single-node obs/info
      key = f"obs:{key}"
    key = key.replace("info:", "obs:")  # merge obs and info together
    key = key.replace("obs:", f"{root_name}:")  # compatible with to_namedtuple
    new_keys.append(key.replace(":", "."))
  dict_tree = to_nested_dict(dict(zip(new_keys, list(range(len(new_keys))))))
  structure = to_namedtuple(root_name, dict_tree)
  paths, indices, treespec = optree.tree_flatten_with_path(structure)
  return paths, indices, treespec


def gym_structure(
  keys: List[str]
) -> Tuple[List[Tuple[str, ...]], List[int], PyTreeSpec]:
  """Convert flat keys into tree structure for dict construction."""
  keys = [k.replace(":", ".") for k in keys]
  dict_tree = to_nested_dict(dict(zip(keys, list(range(len(keys))))))
  paths, indices, treespec = optree.tree_flatten_with_path(dict_tree)
  return paths, indices, treespec


gymnasium_structure = gym_structure
