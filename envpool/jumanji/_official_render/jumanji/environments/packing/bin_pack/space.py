# ruff: noqa
# fmt: off
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
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Space:
    x1: Any
    x2: Any
    y1: Any
    y2: Any
    z1: Any
    z2: Any

    def astype(self, dtype: Any) -> Space:
        space_dict = {key: np.asarray(value, dtype) for key, value in self.__dict__.items()}
        return Space(**space_dict)

    def get_axis_value(self, axis: str, index: int) -> Any:
        """Dynamically returns the correct attribute given the axis ("x", "y" or "z") and the index
        (1 or 2).

        Args:
            axis: string that can either be "x", "y" or "z".
            index: index of attribute to get, either 1 or 2.

        Returns:
            attribute of space matching the axis and index value.

        Example:
            ```python
            space.y2 is space.get_axis_value("y", 2)  # True
            ```
        """
        return getattr(self, f"{axis}{index}")

    def set_axis_value(self, axis: str, index: int, value: Any) -> None:
        """Dynamically sets the correct attribute given the axis ("x", "y" or "z"), the index
        (1 or 2) and the value to set.

        Args:
            axis: string that can either be "x", "y" or "z".
            index: index of attribute to get, either 1 or 2.
            value: value to set the attribute to.

        Example:
            ```python
            space.y2 = 10
            # Using the setter gives:
            space.set_axis_value("y", 2, 10)
            ```
        """
        return setattr(self, f"{axis}{index}", value)

    def __repr__(self) -> str:
        return (
            "Space(\n"
            f"\tx1={self.x1!r}, x2={self.x2!r},\n"
            f"\ty1={self.y1!r}, y2={self.y2!r},\n"
            f"\tz1={self.z1!r}, z2={self.z2!r},\n"
            ")"
        )

    def volume(self) -> Any:
        """Returns the volume as a float to prevent from overflow with 32 bits."""
        x_len = np.asarray(self.x2 - self.x1, np.float32)
        y_len = np.asarray(self.y2 - self.y1, np.float32)
        z_len = np.asarray(self.z2 - self.z1, np.float32)
        return x_len * y_len * z_len

    def intersection(self, space: Space) -> Space:
        """Returns the intersected space with another space (i.e. the space that is included in both
        spaces whose volume is maximum).
        """
        x1 = np.maximum(self.x1, space.x1)
        x2 = np.minimum(self.x2, space.x2)
        y1 = np.maximum(self.y1, space.y1)
        y2 = np.minimum(self.y2, space.y2)
        z1 = np.maximum(self.z1, space.z1)
        z2 = np.minimum(self.z2, space.z2)
        return Space(x1=x1, x2=x2, y1=y1, y2=y2, z1=z1, z2=z2)

    def intersect(self, space: Space) -> Any:
        """Returns whether a space intersect another space or not."""
        return ~(self.intersection(space).is_empty())

    def is_empty(self) -> Any:
        """A space is empty if at least one dimension is negative or zero."""
        return (self.x1 >= self.x2) | (self.y1 >= self.y2) | (self.z1 >= self.z2)

    def is_included(self, space: Space) -> Any:
        """Returns whether self is included into another space."""
        return (
            (self.x1 >= space.x1)
            & (self.x2 <= space.x2)
            & (self.y1 >= space.y1)
            & (self.y2 <= space.y2)
            & (self.z1 >= space.z1)
            & (self.z2 <= space.z2)
        )

    def hyperplane(self, axis: str, direction: str) -> Space:
        """Returns the hyperplane (e.g. lower hyperplane on the x axis) for EMS creation when
        packing an item.

        Args:
            axis: 'x', 'y' or 'z'.
            direction: 'upper' or 'lower'.

        Returns:
            space whose dimensions are all infinite but on the given axis where it is semi-closed.
        """
        inf_ = np.inf
        axis_direction = f"{axis}_{direction}"
        if axis_direction == "x_lower":
            return Space(x1=-inf_, x2=self.x1, y1=-inf_, y2=inf_, z1=-inf_, z2=inf_)
        elif axis_direction == "x_upper":
            return Space(x1=self.x2, x2=inf_, y1=-inf_, y2=inf_, z1=-inf_, z2=inf_)
        elif axis_direction == "y_lower":
            return Space(x1=-inf_, x2=inf_, y1=-inf_, y2=self.y1, z1=-inf_, z2=inf_)
        elif axis_direction == "y_upper":
            return Space(x1=-inf_, x2=inf_, y1=self.y2, y2=inf_, z1=-inf_, z2=inf_)
        elif axis_direction == "z_lower":
            return Space(x1=-inf_, x2=inf_, y1=-inf_, y2=inf_, z1=-inf_, z2=self.z1)
        elif axis_direction == "z_upper":
            return Space(x1=-inf_, x2=inf_, y1=-inf_, y2=inf_, z1=self.z2, z2=inf_)
        else:
            raise ValueError(f"arguments not valid, got axis: {axis} and direction: {direction}.")
