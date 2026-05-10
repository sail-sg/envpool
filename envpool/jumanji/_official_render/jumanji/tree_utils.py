# ruff: noqa
# fmt: off
from __future__ import annotations
# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import fields, is_dataclass, replace
from typing import Sequence, TypeVar

import numpy as np

T = TypeVar("T")


def _tree_map(function, *trees):
    first = trees[0]
    if is_dataclass(first) and not isinstance(first, type):
        return replace(
            first,
            **{
                field.name: _tree_map(
                    function, *(getattr(tree, field.name) for tree in trees)
                )
                for field in fields(first)
            },
        )
    if isinstance(first, tuple) and hasattr(first, "_fields"):
        return type(first)(
            *(_tree_map(function, *values) for values in zip(*trees))
        )
    if isinstance(first, tuple):
        return type(first)(_tree_map(function, *values) for values in zip(*trees))
    if isinstance(first, list):
        return [_tree_map(function, *values) for values in zip(*trees)]
    if isinstance(first, dict):
        return {
            key: _tree_map(function, *(tree[key] for tree in trees))
            for key in first
        }
    return function(*trees)


def tree_transpose(list_of_trees: Sequence[T]) -> T:
    """Convert a list of trees of identical structure into a single tree of arrays.

    Args:
        list_of_trees: list of tree of identical structure.

    Returns:
        tree of arrays.
    """
    return _tree_map(lambda *xs: np.stack(xs, axis=0), *list_of_trees)  # type: ignore


def tree_slice(tree: T, i: Any) -> T:
    """Returns a slice of the tree where all leaves are mapped by x: x[i].

    Args:
        tree: tree of arrays whose ndim is at least 1.
        i: index of the slice.

    Returns:
        tree whose leaves have been reduced to their i-th item
    """
    return _tree_map(lambda x: x[i], tree)  # type: ignore


def tree_add_element(tree: T, i: Any, element: T) -> T:
    """Sets one value of a tree along the batch axis. It is equivalent to

    ```Python
    for array_leaf in tree:
        array_leaf[i] = element[i]
    ```

    Args:
        tree: leaves are arrays and have a batch dimension.
        i: index of arrays to set the value.
        element: pytree with the same structure as `tree`. Its leaves are scalars or arrays whose
            dimension is one less than `tree`.

    Returns:
        tree whose elements are the same as before but with the ith value being set to that of
            the given element.
    """
    def set_element(array, value):
        out = np.array(array, copy=True)
        out[i] = value
        return out

    new_tree: T = _tree_map(set_element, tree, element)
    return new_tree
