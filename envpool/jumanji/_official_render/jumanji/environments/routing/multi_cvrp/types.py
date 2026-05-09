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

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex


@dataclass
class Node:
    """Data structure of a node.
    coordinates: this array stores the x and y coordinates of each customer node.
        The x and y coordinates indicate where the customers are located on the
        map.
    demands: this array stores the capacity demands of each customer. Large
        demands take up more vehicle capacity.
    """

    coordinates: chex.Array  # Shape: (num_customers + 1, 2)
    demands: chex.Array  # Shape: (num_customers + 1,)


@dataclass
class TimeWindow:
    """Tree structure for the time windows.
    start: this array stores the earliest times that the customers
        want their packages to be picked up. If a vehicle arrives before
        this time, a penalty (based on early coeff) will be handed out.
    end: this array stores the latest times that the customers
        want their packages to be picked up. If a vehicle arrives after
        this time, a penalty (based on late coeff) will be handed out.
    """

    start: chex.Array  # Shape: (num_customers,)
    end: chex.Array  # Shape: (num_customers,)


@dataclass
class PenalityCoeff:
    """Tree structure that represents the penalty coefficients.
    early: this array stores the early penalty coefficient values for
        each customer. If a vehicle arrives before the customer's window
        start time it receives
        pentality = (TimeWindow.start - Vehicle.local_times) * PenalityCoeff.early.
    late: this array stores the late penalty coefficient values for
        each customer. If a vehicle arrives after the customer's window
        end time it receives
        pentality = (Vehicle.local_times - TimeWindow.end) * PenalityCoeff.late.
    """

    early: chex.Array  # Shape: (num_customers,)
    late: chex.Array  # Shape: (num_customers,)


@dataclass
class Vehicle:
    """Vehicle tree structure.
    local_times: This array stores the current local times of
        each vehicle. This is necessary as each vehicle has traveled
        a different total distance and therefore has different
        local times for the same environment step.
    capacities: This array stores the capacities of each vehicle. This
        represents the total number of packages/volume that the vehicle can still
        add before needing to return to the depot.
    """

    local_times: chex.Array  # Shape: (num_vehicles,)
    capacities: chex.Array  # Shape: (num_vehicles,)


@dataclass
class ObsVehicle(Vehicle):
    """Vehicle tree structure.
    local_times: This array stores the current local times of
        each vehicle. This is necessary as each vehicle has traveled
        a different total distance and therefore has different
        local times for the same environment step.
    coordinates: This array stores the positions (locations) of each of
        the vehicles.
    capacities: This array stores the capacities of each vehicle. This
        represents the total number of packages/volume that the vehicle can still
        add before needing to return to the depot.
    """

    coordinates: chex.Array  # Shape: (num_vehicles, 2)


@dataclass
class StateVehicle(Vehicle):
    """Vehicle tree structure.
    local_times: This array stores the current local times of
        each vehicle. This is necessary as each vehicle has traveled
        a different total distance and therefore has different
        local times for the same environment step.
    positions: This array stores the positions (locations) of each of
        the vehicles. Here 0 means that the vehicle is at the DEPOT and 1+ that
        it is at a customer.
    capacities: This array stores the capacities of each vehicle. This
        represents the total number of packages/volume that the vehicle can still
        add before needing to return to the depot.
    distances: This array stores the total distances traveled by each vehicle
        thus far.
    time_penalties: This array stores the total time penalties that each vehicle
        has received thus far.
    """

    positions: chex.Array  # Shape: (num_vehicles,)
    distances: chex.Array  # Shape: (num_vehicles,)
    time_penalties: chex.Array  # Shape: (num_vehicles,)


@dataclass
class State:
    """
    nodes: Customer node coordinates and demands.
    windows: Time windows within which a customer should be visited.
    coeffs: Coefficient values used to calculate penalties if the vehicles arrive
        outside their respective time windows.
    vehicles: General information related to each vehicle.
    order: This array stores the history of each vehicle by tracking what customer each
        vehicle was at each environment step. This is used for rendering.
    step_count: The current step count in the environment.
    action_mask: This array stores the marginal action mask for each vehicle.
    key: random key used for auto-reset.
    """

    nodes: Node  # Shape: (num_customers + 1, ...)
    windows: TimeWindow  # Shape: (num_customers, ...)
    coeffs: PenalityCoeff  # Shape: (num_customers, ...)
    vehicles: StateVehicle  # Shape: (num_vehicles, ...)
    order: chex.Array  # Shape: (num_vehicles, 2 * num_customers,) - this size is
    # worst-case when hitting the max step length.
    step_count: chex.Array  # Shape: ()
    action_mask: chex.Array  # Shape: (num_vehicles, num_customers + 1)
    key: chex.PRNGKey  # (2,)


class Observation(NamedTuple):
    """
    Each of these arrays is batched by vehicle id. Therefore one can find an
    individual vehicle's observation by just indexing by that vehicle's id.
    nodes: customer node coordinates and demands.
    windows: time windows within which a customer should be visited.
    coeffs: coefficient values used to calculate penalties if the vehicles arrive
        outside their respective time windows.
    vehicles: this array stores the information of the controllable vehicles.
    action_mask: an array containing the action masks for each vehicle.
    """

    nodes: Node  # Shape: (num_vehicles, num_customers + 1, ...)
    windows: TimeWindow  # Shape: (num_vehicles, num_customers, ...)
    coeffs: PenalityCoeff  # Shape: (num_vehicles, num_customers, ...)
    vehicles: ObsVehicle  # Shape: (num_vehicles, ...)
    action_mask: chex.Array  # Shape: (num_vehicles, num_customers + 1)
