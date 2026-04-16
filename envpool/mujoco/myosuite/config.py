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
"""Metadata-driven MyoSuite config resolution for public registration."""

from __future__ import annotations

import tempfile
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from envpool.mujoco.myosuite.metadata import MYOSUITE_DIRECT_ENTRIES
from envpool.mujoco.myosuite.paths import myosuite_asset_root


def _mujoco() -> Any:
    import mujoco as mujoco_module

    return mujoco_module


def _asset_root(base_path: str) -> Path:
    candidate = Path(base_path) / "mujoco" / "myosuite_assets"
    if candidate.exists():
        return candidate
    return myosuite_asset_root()


def _asset_model_path(base_path: str, model_path: str) -> Path:
    path = Path(model_path)
    if path.is_absolute():
        return path
    return _asset_root(base_path) / model_path


@cache
def _expanded_entry_by_id() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    mapping: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for entry in MYOSUITE_DIRECT_ENTRIES:
        mapping[entry["id"]] = (entry, {})
        for variant_def in entry["variant_defs"]:
            mapping[variant_def["variant_id"]] = (
                entry,
                dict(variant_def["variants"]),
            )
    return mapping


def myosuite_expanded_entry(
    task_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the direct metadata entry plus variant overrides for a task ID."""
    try:
        return _expanded_entry_by_id()[task_id]
    except KeyError as exc:
        raise KeyError(f"Unknown MyoSuite task_id: {task_id}") from exc


@cache
def _model(base_path: str, model_path: str) -> Any:
    mujoco = _mujoco()
    return mujoco.MjModel.from_xml_path(
        str(_asset_model_path(base_path, model_path))
    )


def _replace_all(text: str, old: str, new: str) -> str:
    return text.replace(old, new)


@cache
def _track_model(base_path: str, model_path: str, object_name: str) -> Any:
    mujoco = _mujoco()
    asset_root = _asset_root(base_path)
    source_model = asset_root / model_path
    object_xml = source_model.read_text()
    tabletop_xml = (
        asset_root / "envs/myo/assets/hand/myohand_tabletop.xml"
    ).read_text()
    hand_assets_xml = (
        asset_root / "simhive/myo_sim/hand/assets/myohand_assets.xml"
    ).read_text()
    myo_sim_root = asset_root / "simhive/myo_sim"
    myo_sim_root_str = str(myo_sim_root)

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        hand_assets_tmp = tmp_dir / "myohand_assets.xml"
        tabletop_tmp = tmp_dir / "myohand_tabletop.xml"
        object_tmp = tmp_dir / "myohand_object.xml"

        hand_assets_xml = _replace_all(
            hand_assets_xml,
            'meshdir=".." texturedir=".."',
            f'meshdir="{myo_sim_root_str}" texturedir="{myo_sim_root_str}"',
        )
        hand_assets_tmp.write_text(hand_assets_xml)

        tabletop_xml = _replace_all(
            tabletop_xml,
            "../../../../simhive/myo_sim/hand/assets/myohand_assets.xml",
            str(hand_assets_tmp),
        )
        tabletop_xml = _replace_all(
            tabletop_xml,
            "../../../../simhive/furniture_sim/simpleTable/simpleTable_asset.xml",
            str(
                asset_root
                / "simhive/furniture_sim/simpleTable/simpleTable_asset.xml"
            ),
        )
        tabletop_xml = _replace_all(
            tabletop_xml,
            "../../../../simhive/myo_sim/hand/assets/myohand_body.xml",
            str(asset_root / "simhive/myo_sim/hand/assets/myohand_body.xml"),
        )
        tabletop_xml = _replace_all(
            tabletop_xml,
            "../../../../simhive/furniture_sim/simpleTable/simpleGraniteTable_body.xml",
            str(
                asset_root
                / "simhive/furniture_sim/simpleTable/simpleGraniteTable_body.xml"
            ),
        )
        tabletop_xml = _replace_all(
            tabletop_xml,
            'meshdir="../../../../simhive/myo_sim/" texturedir="../../../../simhive/myo_sim/"',
            f'meshdir="{myo_sim_root_str}" texturedir="{myo_sim_root_str}"',
        )
        tabletop_tmp.write_text(tabletop_xml)

        object_xml = _replace_all(object_xml, "OBJECT_NAME", object_name)
        object_xml = _replace_all(
            object_xml, "myohand_tabletop.xml", str(tabletop_tmp)
        )
        object_xml = _replace_all(
            object_xml,
            "../../../../simhive/object_sim/common.xml",
            str(asset_root / "simhive/object_sim/common.xml"),
        )
        object_xml = _replace_all(
            object_xml,
            f"../../../../simhive/object_sim/{object_name}/assets.xml",
            str(asset_root / f"simhive/object_sim/{object_name}/assets.xml"),
        )
        object_xml = _replace_all(
            object_xml,
            f"../../../../simhive/object_sim/{object_name}/body.xml",
            str(asset_root / f"simhive/object_sim/{object_name}/body.xml"),
        )
        object_tmp.write_text(object_xml)
        return mujoco.MjModel.from_xml_path(str(object_tmp))


def _weighted_reward(kwargs: dict[str, Any], key: str, default: float) -> float:
    return float(kwargs.get("weighted_reward_keys", {}).get(key, default))


def _flatten_site_ranges(
    target_reach_range: dict[str, list[list[float]]],
) -> tuple[list[str], list[float], list[float]]:
    site_names: list[str] = []
    mins: list[float] = []
    maxs: list[float] = []
    for site_name, span in target_reach_range.items():
        site_names.append(site_name)
        mins.extend(span[0])
        maxs.extend(span[1])
    return site_names, mins, maxs


def _dict_reference_init(
    reference: dict[str, Any], key: str, array_key: str
) -> np.ndarray:
    value = reference.get(key)
    if value is not None:
        return np.asarray(value, dtype=np.float64)
    base = np.asarray(reference[array_key], dtype=np.float64)
    if base.shape[0] == 2:
        return np.mean(base, axis=0)
    return base[0]


def _reference_config(base_path: str, reference: Any) -> dict[str, Any]:
    if isinstance(reference, str):
        reference_path = _asset_root(base_path) / reference
        with np.load(reference_path) as data:
            robot = np.asarray(data["robot"], dtype=np.float64)
            obj = np.asarray(data["object"], dtype=np.float64)
            robot_vel = (
                np.asarray(data["robot_vel"], dtype=np.float64)
                if "robot_vel" in data.files
                else None
            )
            robot_init = (
                np.asarray(data["robot_init"], dtype=np.float64)
                if "robot_init" in data.files
                else robot[0]
            )
            object_init = (
                np.asarray(data["object_init"], dtype=np.float64)
                if "object_init" in data.files
                else obj[0]
            )
        return {
            "reference_path": reference,
            "reference_time": [],
            "reference_robot": [],
            "reference_robot_vel": [],
            "reference_object": [],
            "reference_robot_init": [],
            "reference_object_init": [],
            "robot_dim": int(robot.shape[1]),
            "object_dim": int(obj.shape[1]),
            "robot_horizon": int(robot.shape[0]),
            "object_horizon": int(obj.shape[0]),
            "reference_has_robot_vel": robot_vel is not None,
        }

    ref = dict(reference)
    time = np.asarray(ref["time"], dtype=np.float64)
    robot = np.asarray(ref["robot"], dtype=np.float64)
    obj = np.asarray(ref["object"], dtype=np.float64)
    robot_vel = (
        np.asarray(ref["robot_vel"], dtype=np.float64)
        if ref.get("robot_vel") is not None
        else None
    )
    robot_init = _dict_reference_init(ref, "robot_init", "robot")
    object_init = _dict_reference_init(ref, "object_init", "object")
    return {
        "reference_path": "",
        "reference_time": time.tolist(),
        "reference_robot": robot.reshape(-1).tolist(),
        "reference_robot_vel": []
        if robot_vel is None
        else robot_vel.reshape(-1).tolist(),
        "reference_object": obj.reshape(-1).tolist(),
        "reference_robot_init": robot_init.tolist(),
        "reference_object_init": object_init.tolist(),
        "robot_dim": int(robot.shape[1]),
        "object_dim": int(obj.shape[1]),
        "robot_horizon": int(robot.shape[0]),
        "object_horizon": int(obj.shape[0]),
        "reference_has_robot_vel": robot_vel is not None,
    }


def _bimanual_index_sets(model: Any) -> tuple[list[int], ...]:
    myo_qpos: list[int] = []
    myo_dof: list[int] = []
    prosth_qpos: list[int] = []
    prosth_dof: list[int] = []
    manip_joint = model.joint("manip_object/freejoint")
    for joint_id in range(model.njnt):
        joint = model.joint(joint_id)
        if joint.name == "manip_object/freejoint":
            continue
        if joint.name.startswith("prosthesis"):
            prosth_qpos.append(joint.qposadr[0])
            prosth_dof.append(joint.dofadr[0])
        else:
            myo_qpos.append(joint.qposadr[0])
            myo_dof.append(joint.dofadr[0])
    return (
        myo_qpos,
        myo_dof,
        prosth_qpos,
        prosth_dof,
        [manip_joint.qposadr[0] + i for i in range(7)],
        [manip_joint.dofadr[0] + i for i in range(6)],
    )


def _pose_config(
    base_path: str, entry: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    model = _model(base_path, kwargs["model_path"])
    if "target_jnt_range" in kwargs:
        target_qpos_min = [
            bounds[0] for bounds in kwargs["target_jnt_range"].values()
        ]
        target_qpos_max = [
            bounds[1] for bounds in kwargs["target_jnt_range"].values()
        ]
        target_qpos_value = [
            (lo + hi) / 2.0
            for lo, hi in zip(target_qpos_min, target_qpos_max, strict=True)
        ]
    else:
        target_qpos_min = []
        target_qpos_max = []
        target_qpos_value = list(kwargs["target_jnt_value"])
    return {
        "frame_skip": int(kwargs.get("frame_skip", 10)),
        "model_path": str(kwargs["model_path"]),
        "normalize_act": bool(kwargs.get("normalize_act", True)),
        "obs_dim": model.nq + model.nv + model.nq + model.na,
        "qpos_dim": model.nq,
        "qvel_dim": model.nv,
        "act_dim": model.na,
        "action_dim": model.nu,
        "pose_thd": float(kwargs.get("pose_thd", 0.35)),
        "reward_pose_w": _weighted_reward(kwargs, "pose", 1.0),
        "reward_bonus_w": _weighted_reward(kwargs, "bonus", 4.0),
        "reward_act_reg_w": _weighted_reward(kwargs, "act_reg", 1.0),
        "reward_penalty_w": _weighted_reward(kwargs, "penalty", 50.0),
        "reset_type": str(kwargs.get("reset_type", "init")),
        "target_type": str(
            kwargs.get(
                "target_type", "fixed" if target_qpos_value else "generate"
            )
        ),
        "target_qpos_min": target_qpos_min,
        "target_qpos_max": target_qpos_max,
        "target_qpos_value": target_qpos_value,
        "viz_site_targets": list(kwargs.get("viz_site_targets", [])),
        "weight_bodyname": str(kwargs.get("weight_bodyname", "")),
        "weight_range": list(kwargs.get("weight_range", [])),
    }


def _reach_config(
    base_path: str, entry: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    model = _model(base_path, kwargs["model_path"])
    site_names, mins, maxs = _flatten_site_ranges(kwargs["target_reach_range"])
    return {
        "frame_skip": int(kwargs.get("frame_skip", 10)),
        "model_path": str(kwargs["model_path"]),
        "normalize_act": bool(kwargs.get("normalize_act", True)),
        "obs_dim": model.nq + model.nv + 6 * len(site_names) + model.na,
        "qpos_dim": model.nq,
        "qvel_dim": model.nv,
        "act_dim": model.na,
        "action_dim": model.nu,
        "target_site_count": len(site_names),
        "far_th": float(kwargs.get("far_th", 0.35)),
        "reward_reach_w": _weighted_reward(kwargs, "reach", 1.0),
        "reward_bonus_w": _weighted_reward(kwargs, "bonus", 4.0),
        "reward_act_reg_w": _weighted_reward(kwargs, "act_reg", 0.0),
        "reward_penalty_w": _weighted_reward(kwargs, "penalty", 50.0),
        "target_site_names": site_names,
        "target_pos_min": mins,
        "target_pos_max": maxs,
    }


def _myobase_reorient_config(
    base_path: str, entry: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    model = _model(base_path, kwargs["model_path"])
    mode_map = {
        "Geometries100EnvV0": "100",
        "Geometries8EnvV0": "8",
        "InDistribution": "id",
        "OutofDistribution": "ood",
    }
    return {
        "frame_skip": int(kwargs.get("frame_skip", 5)),
        "model_path": str(kwargs["model_path"]),
        "normalize_act": bool(kwargs.get("normalize_act", True)),
        "obs_dim": (model.nq - 6) + 21 + 3 * model.nu + model.na,
        "qpos_dim": model.nq,
        "qvel_dim": model.nv,
        "act_dim": model.na,
        "action_dim": model.nu,
        "randomization_mode": mode_map[entry["class_name"]],
        "reward_pos_align_w": _weighted_reward(kwargs, "pos_align", 1.0),
        "reward_rot_align_w": _weighted_reward(kwargs, "rot_align", 1.0),
        "reward_act_reg_w": _weighted_reward(kwargs, "act_reg", 5.0),
        "reward_drop_w": _weighted_reward(kwargs, "drop", 5.0),
        "reward_bonus_w": _weighted_reward(kwargs, "bonus", 10.0),
    }


def _key_turn_config(
    base_path: str, entry: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    model = _model(base_path, kwargs["model_path"])
    return {
        "frame_skip": int(kwargs.get("frame_skip", 10)),
        "model_path": str(kwargs["model_path"]),
        "normalize_act": bool(kwargs.get("normalize_act", True)),
        "obs_dim": model.nq + model.nv + 6 + model.na,
        "qpos_dim": model.nq,
        "qvel_dim": model.nv,
        "act_dim": model.na,
        "action_dim": model.nu,
        "goal_th": float(kwargs.get("goal_th", np.pi)),
        "reward_key_turn_w": _weighted_reward(kwargs, "key_turn", 1.0),
        "reward_iftip_approach_w": _weighted_reward(
            kwargs, "IFtip_approach", 10.0
        ),
        "reward_thtip_approach_w": _weighted_reward(
            kwargs, "THtip_approach", 10.0
        ),
        "reward_act_reg_w": _weighted_reward(kwargs, "act_reg", 1.0),
        "reward_bonus_w": _weighted_reward(kwargs, "bonus", 4.0),
        "reward_penalty_w": _weighted_reward(kwargs, "penalty", 25.0),
        "key_init_range": list(kwargs.get("key_init_range", (0.0, 0.0))),
    }


def _obj_hold_config(
    base_path: str, entry: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    model = _model(base_path, kwargs["model_path"])
    return {
        "frame_skip": int(kwargs.get("frame_skip", 10)),
        "model_path": str(kwargs["model_path"]),
        "normalize_act": bool(kwargs.get("normalize_act", True)),
        "obs_dim": (model.nq - 7) + (model.nv - 6) + 6 + model.na,
        "qpos_dim": model.nq,
        "qvel_dim": model.nv,
        "act_dim": model.na,
        "action_dim": model.nu,
        "randomize_on_reset": entry["class_name"] == "ObjHoldRandomEnvV0",
        "reward_goal_dist_w": _weighted_reward(kwargs, "goal_dist", 100.0),
        "reward_bonus_w": _weighted_reward(kwargs, "bonus", 4.0),
        "reward_penalty_w": _weighted_reward(kwargs, "penalty", 10.0),
    }


def _torso_config(
    base_path: str, entry: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    model = _model(base_path, kwargs["model_path"])
    target_qpos_value = [
        (bounds[0] + bounds[1]) / 2.0
        for bounds in kwargs["target_jnt_range"].values()
    ]
    pose_dim = len(target_qpos_value)
    return {
        "frame_skip": int(kwargs.get("frame_skip", 5)),
        "model_path": str(kwargs["model_path"]),
        "normalize_act": bool(kwargs.get("normalize_act", True)),
        "obs_dim": model.nq + model.nv + pose_dim + model.na,
        "qpos_dim": model.nq,
        "qvel_dim": model.nv,
        "act_dim": model.na,
        "action_dim": model.nu,
        "pose_dim": pose_dim,
        "pose_thd": float(kwargs.get("pose_thd", 0.25)),
        "reward_pose_w": _weighted_reward(kwargs, "pose", 1.0),
        "reward_bonus_w": _weighted_reward(kwargs, "bonus", 4.0),
        "reward_act_reg_w": _weighted_reward(kwargs, "act_reg", 1.0),
        "reward_penalty_w": _weighted_reward(kwargs, "penalty", 50.0),
        "target_qpos_value": target_qpos_value,
    }


def _pen_twirl_config(
    base_path: str, entry: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    model = _model(base_path, kwargs["model_path"])
    return {
        "frame_skip": int(kwargs.get("frame_skip", 5)),
        "model_path": str(kwargs["model_path"]),
        "normalize_act": bool(kwargs.get("normalize_act", True)),
        "obs_dim": (model.nq - 6) + 21 + model.na,
        "qpos_dim": model.nq,
        "qvel_dim": model.nv,
        "act_dim": model.na,
        "action_dim": model.nu,
        "randomize_target": entry["class_name"] == "PenTwirlRandomEnvV0",
        "reward_pos_align_w": _weighted_reward(kwargs, "pos_align", 1.0),
        "reward_rot_align_w": _weighted_reward(kwargs, "rot_align", 1.0),
        "reward_act_reg_w": _weighted_reward(kwargs, "act_reg", 5.0),
        "reward_drop_w": _weighted_reward(kwargs, "drop", 5.0),
        "reward_bonus_w": _weighted_reward(kwargs, "bonus", 10.0),
    }


def _walk_like_config(
    base_path: str, entry: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    model = _model(base_path, kwargs["model_path"])
    return {
        "frame_skip": int(kwargs.get("frame_skip", 10)),
        "model_path": str(kwargs["model_path"]),
        "normalize_act": bool(kwargs.get("normalize_act", True)),
        "obs_dim": (model.nq - 2)
        + model.nv
        + 2
        + 4
        + 2
        + 1
        + 6
        + 1
        + 3 * model.nu
        + model.na,
        "qpos_dim": model.nq,
        "qvel_dim": model.nv,
        "act_dim": model.na,
        "action_dim": model.nu,
        "min_height": float(kwargs.get("min_height", 0.8)),
        "max_rot": float(kwargs.get("max_rot", 0.8)),
        "hip_period": int(kwargs.get("hip_period", 100)),
        "reset_type": str(kwargs.get("reset_type", "init")),
        "target_x_vel": float(kwargs.get("target_x_vel", 0.0)),
        "target_y_vel": float(kwargs.get("target_y_vel", 1.2)),
        "target_rot": []
        if kwargs.get("target_rot") is None
        else list(kwargs["target_rot"]),
        "terrain": str(kwargs.get("terrain", "")),
        "terrain_variant": ""
        if kwargs.get("variant") is None
        else str(kwargs["variant"]),
        "use_knee_condition": entry["class_name"] == "TerrainEnvV0",
        "reward_vel_w": _weighted_reward(kwargs, "vel_reward", 5.0),
        "reward_done_w": _weighted_reward(kwargs, "done", -100.0),
        "reward_cyclic_hip_w": _weighted_reward(kwargs, "cyclic_hip", -10.0),
        "reward_ref_rot_w": _weighted_reward(kwargs, "ref_rot", 10.0),
        "reward_joint_angle_w": _weighted_reward(
            kwargs, "joint_angle_rew", 5.0
        ),
    }


def _challenge_reorient_config(
    base_path: str, entry: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    model = _model(base_path, kwargs["model_path"])
    obj_mass = kwargs.get("obj_mass_range", [0.108, 0.108])
    return {
        "frame_skip": int(kwargs.get("frame_skip", 5)),
        "model_path": str(kwargs["model_path"]),
        "normalize_act": bool(kwargs.get("normalize_act", True)),
        "obs_dim": (model.nq - 7) + (model.nv - 6) + 18 + model.na,
        "qpos_dim": model.nq,
        "qvel_dim": model.nv,
        "act_dim": model.na,
        "action_dim": model.nu,
        "goal_pos_low": float(kwargs.get("goal_pos", (0.0, 0.0))[0]),
        "goal_pos_high": float(kwargs.get("goal_pos", (0.0, 0.0))[1]),
        "goal_rot_low": float(kwargs.get("goal_rot", (0.0, 0.0))[0]),
        "goal_rot_high": float(kwargs.get("goal_rot", (0.0, 0.0))[1]),
        "obj_size_change": float(kwargs.get("obj_size_change", 0.0)),
        "obj_mass_low": float(obj_mass[0]),
        "obj_mass_high": float(obj_mass[1]),
        "obj_friction_change": list(
            kwargs.get("obj_friction_change", (0.0, 0.0, 0.0))
        ),
        "pos_th": float(kwargs.get("pos_th", 0.025)),
        "rot_th": float(kwargs.get("rot_th", 0.262)),
        "drop_th": float(kwargs.get("drop_th", 0.2)),
        "reward_pos_dist_w": _weighted_reward(kwargs, "pos_dist", 100.0),
        "reward_rot_dist_w": _weighted_reward(kwargs, "rot_dist", 1.0),
        "reward_bonus_w": _weighted_reward(kwargs, "bonus", 0.0),
        "reward_act_reg_w": _weighted_reward(kwargs, "act_reg", 0.0),
        "reward_penalty_w": _weighted_reward(kwargs, "penalty", 0.0),
    }


def _challenge_relocate_config(
    base_path: str, entry: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    model = _model(base_path, kwargs["model_path"])
    target_xyz_range = kwargs["target_xyz_range"]
    target_rot_range = kwargs["target_rxryrz_range"]
    obj_xyz_range = kwargs.get("obj_xyz_range")
    obj_geom_range = kwargs.get("obj_geom_range")
    obj_mass_range = kwargs.get("obj_mass_range")
    obj_friction_range = kwargs.get("obj_friction_range")
    return {
        "frame_skip": int(kwargs.get("frame_skip", 5)),
        "model_path": str(kwargs["model_path"]),
        "normalize_act": bool(kwargs.get("normalize_act", True)),
        "obs_dim": (model.nq - 7) + (model.nv - 6) + 18 + model.na,
        "qpos_dim": model.nq,
        "qvel_dim": model.nv,
        "act_dim": model.na,
        "action_dim": model.nu,
        "target_xyz_low": list(target_xyz_range["low"]),
        "target_xyz_high": list(target_xyz_range["high"]),
        "target_rxryrz_low": list(target_rot_range["low"]),
        "target_rxryrz_high": list(target_rot_range["high"]),
        "obj_xyz_low": []
        if obj_xyz_range is None
        else list(obj_xyz_range["low"]),
        "obj_xyz_high": []
        if obj_xyz_range is None
        else list(obj_xyz_range["high"]),
        "obj_geom_low": []
        if obj_geom_range is None
        else list(obj_geom_range["low"]),
        "obj_geom_high": []
        if obj_geom_range is None
        else list(obj_geom_range["high"]),
        "obj_mass_low": 0.0
        if obj_mass_range is None
        else float(obj_mass_range["low"]),
        "obj_mass_high": 0.0
        if obj_mass_range is None
        else float(obj_mass_range["high"]),
        "obj_friction_low": []
        if obj_friction_range is None
        else list(obj_friction_range["low"]),
        "obj_friction_high": []
        if obj_friction_range is None
        else list(obj_friction_range["high"]),
        "qpos_noise_range": float(kwargs.get("qpos_noise_range", 0.0)),
        "pos_th": float(kwargs.get("pos_th", 0.025)),
        "rot_th": float(kwargs.get("rot_th", 0.262)),
        "drop_th": float(kwargs.get("drop_th", 0.5)),
        "reward_pos_dist_w": _weighted_reward(kwargs, "pos_dist", 100.0),
        "reward_rot_dist_w": _weighted_reward(kwargs, "rot_dist", 1.0),
        "reward_act_reg_w": _weighted_reward(kwargs, "act_reg", 0.0),
    }


def _baoding_config(
    base_path: str, entry: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    mujoco = _mujoco()
    model = _model(base_path, kwargs["model_path"])
    ball1_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball1")
    obj_size_range = kwargs.get("obj_size_range")
    obj_mass_range = kwargs.get("obj_mass_range")
    obj_friction_change = kwargs.get("obj_friction_change")
    return {
        "frame_skip": int(kwargs.get("frame_skip", 10)),
        "model_path": str(kwargs["model_path"]),
        "normalize_act": bool(kwargs.get("normalize_act", True)),
        "obs_dim": (model.nq - 14) + 24 + model.na,
        "qpos_dim": model.nq,
        "qvel_dim": model.nv,
        "act_dim": model.na,
        "action_dim": model.nu,
        "drop_th": float(kwargs.get("drop_th", 1.25)),
        "proximity_th": float(kwargs.get("proximity_th", 0.015)),
        "goal_time_period_low": float(
            kwargs.get("goal_time_period", (5, 5))[0]
        ),
        "goal_time_period_high": float(
            kwargs.get("goal_time_period", (5, 5))[1]
        ),
        "goal_xrange_low": float(kwargs.get("goal_xrange", (0.025, 0.025))[0]),
        "goal_xrange_high": float(kwargs.get("goal_xrange", (0.025, 0.025))[1]),
        "goal_yrange_low": float(kwargs.get("goal_yrange", (0.028, 0.028))[0]),
        "goal_yrange_high": float(kwargs.get("goal_yrange", (0.028, 0.028))[1]),
        "task_choice": str(kwargs.get("task_choice", "fixed")),
        "fixed_task": 2,
        "reward_pos_dist_1_w": _weighted_reward(kwargs, "pos_dist_1", 5.0),
        "reward_pos_dist_2_w": _weighted_reward(kwargs, "pos_dist_2", 5.0),
        "obj_size_low": 0.0
        if obj_size_range is None
        else float(obj_size_range[0]),
        "obj_size_high": 0.0
        if obj_size_range is None
        else float(obj_size_range[1]),
        "obj_mass_low": 0.0
        if obj_mass_range is None
        else float(obj_mass_range[0]),
        "obj_mass_high": 0.0
        if obj_mass_range is None
        else float(obj_mass_range[1]),
        "obj_friction_low": []
        if obj_friction_change is None
        else list(
            (
                model.geom_friction[ball1_gid] - np.asarray(obj_friction_change)
            ).tolist()
        ),
        "obj_friction_high": []
        if obj_friction_change is None
        else list(
            (
                model.geom_friction[ball1_gid] + np.asarray(obj_friction_change)
            ).tolist()
        ),
    }


def _bimanual_config(
    base_path: str, entry: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    model = _model(base_path, kwargs["model_path"])
    myo_qpos, myo_dof, prosth_qpos, prosth_dof, _, _ = _bimanual_index_sets(
        model
    )
    obj_bid = model.body("manip_object").id
    obj_gid = model.body(obj_bid).geomadr + 1
    base_friction = np.asarray(model.geom_friction[obj_gid]).reshape(-1)
    obj_mass_change = kwargs.get("obj_mass_change")
    obj_friction_change = kwargs.get("obj_friction_change")
    return {
        "frame_skip": int(kwargs.get("frame_skip", 5)),
        "model_path": str(kwargs["model_path"]),
        "normalize_act": bool(kwargs.get("normalize_act", True)),
        "obs_dim": 1
        + len(myo_qpos)
        + len(myo_dof)
        + len(prosth_qpos)
        + len(prosth_dof)
        + 7
        + 6
        + 5
        + model.na,
        "qpos_dim": model.nq,
        "qvel_dim": model.nv,
        "act_dim": model.na,
        "action_dim": model.nu,
        "proximity_th": 0.17,
        "start_center": [-0.4, -0.25, 1.05],
        "goal_center": [0.4, -0.25, 1.05],
        "start_shifts": [0.055, 0.055, 0.0],
        "goal_shifts": [0.098, 0.098, 0.0],
        "reward_reach_dist_w": _weighted_reward(kwargs, "reach_dist", -0.1),
        "reward_act_w": _weighted_reward(kwargs, "act", 0.0),
        "reward_fin_dis_w": _weighted_reward(kwargs, "fin_dis", -0.5),
        "reward_pass_err_w": _weighted_reward(kwargs, "pass_err", -1.0),
        "obj_scale_change": list(kwargs.get("obj_scale_change", [])),
        "obj_mass_low": 0.0
        if obj_mass_change is None
        else float(model.body_mass[obj_bid] + obj_mass_change[0]),
        "obj_mass_high": 0.0
        if obj_mass_change is None
        else float(model.body_mass[obj_bid] + obj_mass_change[1]),
        "obj_friction_low": []
        if obj_friction_change is None
        else list((base_friction - np.asarray(obj_friction_change)).tolist()),
        "obj_friction_high": []
        if obj_friction_change is None
        else list((base_friction + np.asarray(obj_friction_change)).tolist()),
    }


def _runtrack_config(
    base_path: str, entry: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    model = _model(base_path, kwargs["model_path"])
    return {
        "frame_skip": int(kwargs.get("frame_skip", 5)),
        "model_path": str(kwargs["model_path"]),
        "normalize_act": bool(kwargs.get("normalize_act", True)),
        "obs_dim": 17 + 17 + 2 + 4 + model.na * 4 + 2 + 2,
        "qpos_dim": model.nq,
        "qvel_dim": model.nv,
        "act_dim": model.na,
        "action_dim": model.na,
        "ctrl_dim": model.nu,
        "reset_type": str(kwargs.get("reset_type", "random")),
        "terrain": str(kwargs.get("terrain", "flat")),
        "start_pos": float(kwargs.get("start_pos", 14)),
        "end_pos": float(kwargs.get("end_pos", -15)),
        "real_width": float(kwargs.get("real_width", 1)),
        "reward_sparse_w": 1.0,
        "reward_solved_w": 10.0,
    }


def _soccer_config(
    base_path: str, entry: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    mujoco = _mujoco()
    model = _model(base_path, kwargs["model_path"])
    internal_joint_count = sum(
        1
        for joint_id in range(model.njnt)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        not in (None, "root")
    )
    probabilities = kwargs.get("goalkeeper_probabilities", [0.1, 0.45, 0.45])
    return {
        "frame_skip": int(kwargs.get("frame_skip", 10)),
        "model_path": str(kwargs["model_path"]),
        "normalize_act": bool(kwargs.get("normalize_act", True)),
        "obs_dim": 1
        + internal_joint_count
        + internal_joint_count
        + 4
        + 4
        + model.na * 4
        + 3
        + 12
        + 7
        + 6
        + 2,
        "qpos_dim": model.nq,
        "qvel_dim": model.nv,
        "act_dim": model.na,
        "action_dim": model.nu,
        "reset_type": str(kwargs.get("reset_type", "none")),
        "min_agent_spawn_distance": float(
            kwargs.get("min_agent_spawn_distance", 1)
        ),
        "random_vel_low": float(kwargs.get("random_vel_range", [1.0, 5.0])[0]),
        "random_vel_high": float(kwargs.get("random_vel_range", [1.0, 5.0])[1]),
        "rnd_pos_noise": float(kwargs.get("rnd_pos_noise", 1.0)),
        "rnd_joint_noise": float(kwargs.get("rnd_joint_noise", 0.02)),
        "goalkeeper_probabilities": list(probabilities),
        "max_time_sec": float(kwargs.get("max_time_sec", 10)),
        "reward_goal_scored_w": 1000.0,
        "reward_time_cost_w": -0.01,
        "reward_act_reg_w": -100.0,
        "reward_pain_w": -10.0,
    }


def _chasetag_config(
    base_path: str, entry: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    model = _model(base_path, kwargs["model_path"])
    return {
        "frame_skip": int(kwargs.get("frame_skip", 10)),
        "model_path": str(kwargs["model_path"]),
        "normalize_act": bool(kwargs.get("normalize_act", True)),
        "obs_dim": 28 + 28 + 4 + 4 + 3 + 2 + 2 + 2 + model.na * 4,
        "qpos_dim": model.nq,
        "qvel_dim": model.nv,
        "act_dim": model.na,
        "action_dim": model.nu,
        "reset_type": str(kwargs.get("reset_type", "init")),
        "win_distance": float(kwargs.get("win_distance", 0.5)),
        "min_spawn_distance": float(kwargs.get("min_spawn_distance", 2)),
        "task_choice": str(kwargs.get("task_choice", "CHASE")),
        "terrain": str(kwargs.get("terrain", "FLAT")),
        "repeller_opponent": bool(kwargs.get("repeller_opponent", False)),
        "chase_vel_low": float(kwargs.get("chase_vel_range", [1.0, 1.0])[0]),
        "chase_vel_high": float(kwargs.get("chase_vel_range", [1.0, 1.0])[1]),
        "random_vel_low": float(kwargs.get("random_vel_range", [-2.0, 2.0])[0]),
        "random_vel_high": float(
            kwargs.get("random_vel_range", [-2.0, 2.0])[1]
        ),
        "repeller_vel_low": float(
            kwargs.get("repeller_vel_range", [0.3, 1.0])[0]
        ),
        "repeller_vel_high": float(
            kwargs.get("repeller_vel_range", [0.3, 1.0])[1]
        ),
        "opponent_probabilities": list(
            kwargs.get("opponent_probabilities", [0.1, 0.45, 0.45])
        ),
        "reward_distance_w": -0.1,
        "reward_lose_w": -1000.0,
    }


def _tabletennis_config(
    base_path: str, entry: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    mujoco = _mujoco()
    model = _model(base_path, kwargs["model_path"])
    body_joint_count = sum(
        1
        for joint_id in range(model.njnt)
        if (
            (
                name := mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                )
            )
            is not None
            and not name.startswith("ping")
            and name != "paddle_freejoint"
        )
    )
    ball_xyz_range = kwargs.get("ball_xyz_range") or {}
    ball_friction_range = kwargs.get("ball_friction_range") or {}
    paddle_mass_range = kwargs.get("paddle_mass_range") or [0.0, 0.0]
    qpos_noise_range = kwargs.get("qpos_noise_range")
    qpos_noise_low = float("nan")
    qpos_noise_high = float("nan")
    if qpos_noise_range is not None:
        qpos_noise_low = -float(qpos_noise_range)
        qpos_noise_high = float(qpos_noise_range)
    return {
        "frame_skip": int(kwargs.get("frame_skip", 5)),
        "model_path": str(kwargs["model_path"]),
        "normalize_act": bool(kwargs.get("normalize_act", True)),
        "obs_dim": 1
        + 3
        + body_joint_count
        + body_joint_count
        + 3
        + 3
        + 3
        + 3
        + 4
        + 3
        + 6
        + model.na,
        "qpos_dim": model.nq,
        "qvel_dim": model.nv,
        "act_dim": model.na,
        "action_dim": model.nu,
        "ball_xyz_low": list(ball_xyz_range.get("low", [])),
        "ball_xyz_high": list(ball_xyz_range.get("high", [])),
        "ball_qvel": bool(kwargs.get("ball_qvel", False)),
        "ball_friction_low": list(ball_friction_range.get("low", [])),
        "ball_friction_high": list(ball_friction_range.get("high", [])),
        "paddle_mass_low": float(paddle_mass_range[0]),
        "paddle_mass_high": float(paddle_mass_range[1]),
        "qpos_noise_low": qpos_noise_low,
        "qpos_noise_high": qpos_noise_high,
        "rally_count": int(kwargs.get("rally_count", 1)),
        "reward_reach_dist_w": 1.0,
        "reward_palm_dist_w": 1.0,
        "reward_paddle_quat_w": 2.0,
        "reward_act_reg_w": 0.5,
        "reward_torso_up_w": 2.0,
        "reward_sparse_w": 100.0,
        "reward_solved_w": 1000.0,
        "reward_done_w": -10.0,
    }


def _track_config(
    base_path: str, entry: dict[str, Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    model = _track_model(base_path, kwargs["model_path"], kwargs["object_name"])
    reference = _reference_config(base_path, kwargs["reference"])
    obs_dim = (
        model.nq
        + model.nv
        + reference["robot_dim"]
        + (
            reference["robot_dim"]
            if reference["reference_has_robot_vel"]
            else 1
        )
        + 3
        + model.na
    )
    return {
        "frame_skip": int(kwargs.get("frame_skip", 10)),
        "model_path": str(kwargs["model_path"]),
        "object_name": str(kwargs["object_name"]),
        "reference_path": str(reference["reference_path"]),
        "reference_time": list(reference["reference_time"]),
        "reference_robot": list(reference["reference_robot"]),
        "reference_robot_vel": list(reference["reference_robot_vel"]),
        "reference_object": list(reference["reference_object"]),
        "reference_robot_init": list(reference["reference_robot_init"]),
        "reference_object_init": list(reference["reference_object_init"]),
        "normalize_act": bool(kwargs.get("normalize_act", True)),
        "obs_dim": obs_dim,
        "qpos_dim": model.nq,
        "qvel_dim": model.nv,
        "act_dim": model.na,
        "action_dim": model.nu,
        "robot_dim": reference["robot_dim"],
        "object_dim": reference["object_dim"],
        "robot_horizon": reference["robot_horizon"],
        "object_horizon": reference["object_horizon"],
        "reference_has_robot_vel": bool(reference["reference_has_robot_vel"]),
        "motion_start_time": float(kwargs.get("motion_start_time", 0.0)),
        "motion_extrapolation": bool(kwargs.get("motion_extrapolation", True)),
        "reward_pose_w": _weighted_reward(kwargs, "pose", 0.0),
        "reward_object_w": _weighted_reward(kwargs, "object", 1.0),
        "reward_bonus_w": _weighted_reward(kwargs, "bonus", 1.0),
        "reward_penalty_w": _weighted_reward(kwargs, "penalty", -2.0),
        "terminate_obj_fail": bool(
            kwargs.get(
                "Termimate_obj_fail", kwargs.get("terminate_obj_fail", True)
            )
        ),
        "terminate_pose_fail": bool(
            kwargs.get(
                "Termimate_pose_fail", kwargs.get("terminate_pose_fail", False)
            )
        ),
    }


def _resolve_dynamic_myosuite_task_config(
    entry: dict[str, Any],
    variant_kwargs: dict[str, Any],
    preview_kwargs: dict[str, Any],
) -> dict[str, Any]:
    kwargs = {**entry["kwargs"], **variant_kwargs, **preview_kwargs}
    base_path = str(preview_kwargs["base_path"])
    class_name = entry["class_name"]
    suite = entry["suite"]

    if class_name == "PoseEnvV0":
        config = _pose_config(base_path, entry, kwargs)
    elif class_name == "ReachEnvV0":
        config = _reach_config(base_path, entry, kwargs)
    elif class_name in {
        "Geometries100EnvV0",
        "Geometries8EnvV0",
        "InDistribution",
        "OutofDistribution",
    }:
        config = _myobase_reorient_config(base_path, entry, kwargs)
    elif class_name == "KeyTurnEnvV0":
        config = _key_turn_config(base_path, entry, kwargs)
    elif class_name in {"ObjHoldFixedEnvV0", "ObjHoldRandomEnvV0"}:
        config = _obj_hold_config(base_path, entry, kwargs)
    elif class_name == "TorsoEnvV0":
        config = _torso_config(base_path, entry, kwargs)
    elif class_name in {"PenTwirlFixedEnvV0", "PenTwirlRandomEnvV0"}:
        config = _pen_twirl_config(base_path, entry, kwargs)
    elif class_name in {"WalkEnvV0", "TerrainEnvV0"}:
        config = _walk_like_config(base_path, entry, kwargs)
    elif suite == "myochallenge" and class_name == "ReorientEnvV0":
        config = _challenge_reorient_config(base_path, entry, kwargs)
    elif class_name == "RelocateEnvV0":
        config = _challenge_relocate_config(base_path, entry, kwargs)
    elif class_name == "BaodingEnvV1":
        config = _baoding_config(base_path, entry, kwargs)
    elif class_name == "BimanualEnvV1":
        config = _bimanual_config(base_path, entry, kwargs)
    elif class_name == "RunTrack":
        config = _runtrack_config(base_path, entry, kwargs)
    elif class_name == "SoccerEnvV0":
        config = _soccer_config(base_path, entry, kwargs)
    elif class_name == "ChaseTagEnvV0":
        config = _chasetag_config(base_path, entry, kwargs)
    elif class_name == "TableTennisEnvV0":
        config = _tabletennis_config(base_path, entry, kwargs)
    elif class_name == "TrackEnv":
        config = _track_config(base_path, entry, kwargs)
    else:
        raise ValueError(f"Unsupported MyoSuite class_name: {class_name}")

    if "muscle_condition" in kwargs:
        config["muscle_condition"] = str(kwargs["muscle_condition"])
    return config


def generate_myosuite_task_config(
    entry: dict[str, Any],
    variant_kwargs: dict[str, Any],
    *,
    base_path: str,
) -> dict[str, Any]:
    """Generate a canonical task config for vendored metadata snapshots."""
    return _resolve_dynamic_myosuite_task_config(
        entry,
        variant_kwargs,
        {"base_path": base_path},
    )


def _has_dynamic_task_overrides(
    entry: dict[str, Any],
    variant_kwargs: dict[str, Any],
    preview_kwargs: dict[str, Any],
) -> bool:
    defaults = {**entry["kwargs"], **variant_kwargs}
    for key, value in preview_kwargs.items():
        if key == "base_path" or key.startswith("test_"):
            continue
        if key in defaults and defaults[key] != value:
            return True
    return False


def resolve_myosuite_task_config(
    task_id: str, preview_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Resolve a public MyoSuite task ID into spec kwargs for registration."""
    entry, variant_kwargs = myosuite_expanded_entry(task_id)
    if _has_dynamic_task_overrides(entry, variant_kwargs, preview_kwargs):
        return _resolve_dynamic_myosuite_task_config(
            entry, variant_kwargs, preview_kwargs
        )

    default_config = entry.get("default_config")
    if default_config is None:
        return _resolve_dynamic_myosuite_task_config(
            entry, variant_kwargs, preview_kwargs
        )

    config = dict(default_config)
    muscle_condition = preview_kwargs.get(
        "muscle_condition",
        variant_kwargs.get("muscle_condition"),
    )
    if muscle_condition is not None:
        config["muscle_condition"] = str(muscle_condition)
    return config
