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
"""Native hidden-state transfer for MyoSuite validation and doc images."""

from collections.abc import Mapping
from typing import Any

import numpy as np

_SYNC_STATE_SIZES = {
    "target_jnt_value": "nq",
    "baoding_goal_parameters": "five",
    "chase_task": "one",
    "npc_parameters": "three",
    "qpos0": "nq",
    "qvel0": "nv",
    "act0": "na",
    "qacc0": "nv",
    "qacc_warmstart0": "nv",
    "ctrl": "nu",
    "site_pos": "nsite3",
    "site_quat": "nsite4",
    "site_size": "nsite3",
    "site_rgba": "nsite4",
    "body_pos": "nbody3",
    "body_quat": "nbody4",
    "body_mass": "nbody",
    "geom_pos": "ngeom3",
    "geom_quat": "ngeom4",
    "geom_size": "ngeom3",
    "geom_rgba": "ngeom4",
    "geom_friction": "ngeom3",
    "geom_aabb": "ngeom6",
    "geom_rbound": "ngeom",
    "geom_contype": "ngeom",
    "geom_conaffinity": "ngeom",
    "geom_type": "ngeom",
    "geom_condim": "ngeom",
    "mesh_polymap": "nmeshpolymap",
    "mesh_polynormal": "nmeshpoly3",
    "mesh_polyvert": "nmeshpolyvert",
    "mesh_polyvertadr": "nmeshpoly",
    "mesh_polyvertnum": "nmeshpoly",
    "hfield_data": "nhfielddata",
    "mocap_pos": "nmocap3",
    "mocap_quat": "nmocap4",
    "fatigue_ma": "nu",
    "fatigue_mr": "nu",
    "fatigue_mf": "nu",
    "fatigue_tl": "nu",
}


def sync_state_from_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Trim native reset state arrays for the official oracle."""
    dims = {
        "one": 1,
        "five": 5,
        "three": 3,
        "nq": int(np.asarray(info["model_nq"]).ravel()[0]),
        "nv": int(np.asarray(info["model_nv"]).ravel()[0]),
        "na": int(np.asarray(info["model_na"]).ravel()[0]),
        "nu": int(np.asarray(info["model_nu"]).ravel()[0]),
        "nsite": int(np.asarray(info["model_nsite"]).ravel()[0]),
        "nbody": int(np.asarray(info["model_nbody"]).ravel()[0]),
        "ngeom": int(np.asarray(info["model_ngeom"]).ravel()[0]),
        "nmeshpoly": int(np.asarray(info["model_nmeshpoly"]).ravel()[0]),
        "nmeshpolyvert": int(
            np.asarray(info["model_nmeshpolyvert"]).ravel()[0]
        ),
        "nmeshpolymap": int(np.asarray(info["model_nmeshpolymap"]).ravel()[0]),
        "nhfielddata": int(np.asarray(info["model_nhfielddata"]).ravel()[0]),
        "nmocap": int(np.asarray(info["model_nmocap"]).ravel()[0]),
    }
    dims.update({
        "nmeshpoly3": dims["nmeshpoly"] * 3,
        "nsite3": dims["nsite"] * 3,
        "nsite4": dims["nsite"] * 4,
        "nbody3": dims["nbody"] * 3,
        "nbody4": dims["nbody"] * 4,
        "ngeom3": dims["ngeom"] * 3,
        "ngeom4": dims["ngeom"] * 4,
        "ngeom6": dims["ngeom"] * 6,
        "nmocap3": dims["nmocap"] * 3,
        "nmocap4": dims["nmocap"] * 4,
    })
    sync_state = {}
    for key, dimension in _SYNC_STATE_SIZES.items():
        if key not in info:
            continue
        size = dims[dimension]
        sync_state[key] = np.asarray(info[key][0]).ravel()[:size].tolist()
    sampled = sampled_state_from_info(info)
    sampled.pop("npc_velocity", None)  # Reset precedes the first NPC command.
    return dict(sync_state, **sampled)


def sampled_state_from_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Export sampled goals/commands, without copying evolved physics state."""
    state = {}
    if int(info["npc_parameters"][0, 0]) in (2, 3):
        state["npc_velocity"] = info["npc_velocity"][0].tolist()
    reference = info["random_reference"][0]
    if reference.size > 1:
        state["random_reference"] = reference.tolist()
    return state


def physics_state_from_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Current native integrator state, before the next transition is tested."""
    return {
        key: np
        .asarray(info[key][0])
        .ravel()[: int(np.asarray(info[dimension]).ravel()[0])]
        .tolist()
        for key, dimension in (
            ("qpos", "model_nq"),
            ("qvel", "model_nv"),
            ("act", "model_na"),
            ("qacc_warmstart", "model_nv"),
        )
    }
