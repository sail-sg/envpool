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

from typing import TYPE_CHECKING, NamedTuple, Optional

from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex
import jax.numpy as jnp

from jumanji.environments.packing.bin_pack.space import Space

Container: TypeAlias = Space
EMS: TypeAlias = Space


def empty_ems() -> EMS:
    """Returns an empty EMS located at (0, 0, 0)."""
    return EMS(x1=0, x2=0, y1=0, y2=0, z1=0, z2=0).astype(jnp.int32)


class Item(NamedTuple):
    x_len: chex.Numeric
    y_len: chex.Numeric
    z_len: chex.Numeric


def item_from_space(space: Space) -> Item:
    """Convert a space to an item whose length on each dimension is the length of the space."""
    return Item(
        x_len=space.x2 - space.x1,
        y_len=space.y2 - space.y1,
        z_len=space.z2 - space.z1,
    )


def item_fits_in_item(item: Item, other_item: Item) -> chex.Array:
    """Check if an item is smaller than another one."""
    return (
        (item.x_len <= other_item.x_len)
        & (item.y_len <= other_item.y_len)
        & (item.z_len <= other_item.z_len)
    )


def item_volume(item: Item) -> chex.Array:
    """Returns the volume as a float to prevent from overflow with 32 bits."""
    x_len = jnp.asarray(item.x_len, float)
    y_len = jnp.asarray(item.y_len, float)
    z_len = jnp.asarray(item.z_len, float)
    return x_len * y_len * z_len


class Location(NamedTuple):
    x: chex.Numeric
    y: chex.Numeric
    z: chex.Numeric


def location_from_space(space: Space) -> Location:
    """Returns the location of a space, i.e. the coordinates of its bottom left corner.

    Args:
        space: space object from which to get the location.

    Returns:
        location of the space object (x1, y1, z1).

    """
    return Location(
        x=space.x1,
        y=space.y1,
        z=space.z1,
    )


def space_from_item_and_location(item: Item, location: Location) -> Space:
    """Returns a space from an item at a particular location. The bottom left corner is given
    by the location while the top right is the location plus the item dimensions.
    """
    return Space(
        x1=location.x,
        x2=location.x + item.x_len,
        y1=location.y,
        y2=location.y + item.y_len,
        z1=location.z,
        z2=location.z + item.z_len,
    )


@dataclass
class State:
    """
    container: space defined by 2 points, i.e. 6 coordinates.
    ems: empty maximal spaces (EMSs) in the container, each defined by 2 points (6 coordinates).
    ems_mask: array of booleans that indicate the EMSs that are valid.
    items: defined by 3 attributes (x, y, z).
    items_mask: array of booleans that indicate the items that can be packed.
    items_placed: array of booleans that indicate the items that have been placed so far.
    items_location: locations of items in the container, defined by 3 coordinates (x, y, x).
    action_mask: array of booleans that indicate the valid actions, i.e. EMSs and items that can
        be chosen.
    sorted_ems_indexes: EMS indexes that are sorted by decreasing volume order.
    key: random key used for auto-reset.
    """

    container: Container  # leaves of shape ()
    ems: EMS  # leaves of shape (max_num_ems,)
    ems_mask: chex.Array  # (max_num_ems,)
    items: Item  # leaves of shape (max_num_items,)
    items_mask: chex.Array  # (max_num_items,)
    items_placed: chex.Array  # (max_num_items,)
    items_location: Location  # leaves of shape (max_num_items,)
    action_mask: Optional[chex.Array]  # (obs_num_ems, max_num_items)
    sorted_ems_indexes: chex.Array  # (max_num_ems,)
    key: chex.PRNGKey  # (2,)


class Observation(NamedTuple):
    """
    ems: empty maximal spaces (EMSs) in the container, defined by 2 points (6 coordinates).
    ems_mask: array of booleans that indicate the EMSs that are valid.
    items: defined by 3 attributes (x, y, z).
    items_mask: array of booleans that indicate the items that are valid.
    items_placed: array of booleans that indicate the items that have been placed so far.
    action_mask: array of booleans that indicate the feasible actions, i.e. EMSs and items that can
        be chosen.
    """

    ems: EMS  # leaves of shape (obs_num_ems,)
    ems_mask: chex.Array  # (obs_num_ems,)
    items: Item  # leaves of shape (max_num_items,)
    items_mask: chex.Array  # (max_num_items,)
    items_placed: chex.Array  # (max_num_items,)
    action_mask: chex.Array  # (obs_num_ems, max_num_items)
