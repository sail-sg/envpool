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

from typing import TYPE_CHECKING, NamedTuple

import chex

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class State:
    """
    weights: array of weights of the items.
    values: array of values of the items.
    packed_items: binary mask denoting which items are already packed into the knapsack.
    remaining_budget: the budget currently remaining.
    key: random key used for auto-reset.
    """

    weights: chex.Array  # (num_items,)
    values: chex.Array  # (num_items,)
    packed_items: chex.Array  # (num_items,)
    remaining_budget: chex.Array  # ()
    key: chex.PRNGKey  # (2,)


class Observation(NamedTuple):
    """
    weights: array of weights of the items.
    values: array of values of the items.
    packed_items: binary mask denoting which items are already packed into the knapsack.
    action_mask: binary mask denoting which items can be packed into the knapsack.
    """

    weights: chex.Array  # (num_items,)
    values: chex.Array  # (num_items,)
    packed_items: chex.Array  # (num_items,)
    action_mask: chex.Array  # (num_items,)
