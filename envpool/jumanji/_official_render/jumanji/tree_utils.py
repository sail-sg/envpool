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

from typing import Sequence, TypeVar

import chex
import jax
import jax.numpy as jnp

T = TypeVar("T")


def tree_transpose(list_of_trees: Sequence[T]) -> T:
    """Convert a list of trees of identical structure into a single tree of arrays.

    Args:
        list_of_trees: list of tree of identical structure.

    Returns:
        tree of arrays.
    """
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *list_of_trees)  # type: ignore


def tree_slice(tree: T, i: chex.Numeric) -> T:
    """Returns a slice of the tree where all leaves are mapped by x: x[i].

    Args:
        tree: tree of arrays whose ndim is at least 1.
        i: index of the slice.

    Returns:
        tree whose leaves have been reduced to their i-th item
    """
    return jax.tree_util.tree_map(lambda x: x[i], tree)  # type: ignore


def tree_add_element(tree: T, i: chex.Numeric, element: T) -> T:
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
    new_tree: T = jax.tree_util.tree_map(lambda array, value: array.at[i].set(value), tree, element)
    return new_tree
