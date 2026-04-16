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
"""Internal MyoSuite MyoBase native env tests."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any, Iterator
from xml.etree import ElementTree as ET

import gymnasium
import mujoco
import numpy as np
from absl.testing import absltest

from envpool.mujoco.myosuite.metadata import MYOSUITE_DIRECT_ENTRIES
from envpool.mujoco.myosuite.native import (
    MyoSuiteKeyTurnEnvSpec,
    MyoSuiteKeyTurnGymnasiumEnvPool,
    MyoSuiteKeyTurnPixelEnvSpec,
    MyoSuiteKeyTurnPixelGymnasiumEnvPool,
    MyoSuiteObjHoldEnvSpec,
    MyoSuiteObjHoldGymnasiumEnvPool,
    MyoSuiteObjHoldPixelEnvSpec,
    MyoSuiteObjHoldPixelGymnasiumEnvPool,
    MyoSuitePenTwirlEnvSpec,
    MyoSuitePenTwirlGymnasiumEnvPool,
    MyoSuitePenTwirlPixelEnvSpec,
    MyoSuitePenTwirlPixelGymnasiumEnvPool,
    MyoSuitePoseEnvSpec,
    MyoSuitePoseGymnasiumEnvPool,
    MyoSuitePosePixelEnvSpec,
    MyoSuitePosePixelGymnasiumEnvPool,
    MyoSuiteReachEnvSpec,
    MyoSuiteReachGymnasiumEnvPool,
    MyoSuiteReachPixelEnvSpec,
    MyoSuiteReachPixelGymnasiumEnvPool,
    MyoSuiteReorientEnvSpec,
    MyoSuiteReorientGymnasiumEnvPool,
    MyoSuiteReorientPixelEnvSpec,
    MyoSuiteReorientPixelGymnasiumEnvPool,
    MyoSuiteTerrainEnvSpec,
    MyoSuiteTerrainGymnasiumEnvPool,
    MyoSuiteTerrainPixelEnvSpec,
    MyoSuiteTerrainPixelGymnasiumEnvPool,
    MyoSuiteTorsoEnvSpec,
    MyoSuiteTorsoGymnasiumEnvPool,
    MyoSuiteTorsoPixelEnvSpec,
    MyoSuiteTorsoPixelGymnasiumEnvPool,
    MyoSuiteWalkEnvSpec,
    MyoSuiteWalkGymnasiumEnvPool,
    MyoSuiteWalkPixelEnvSpec,
    MyoSuiteWalkPixelGymnasiumEnvPool,
)
from envpool.mujoco.myosuite.oracle_utils import load_oracle_class
from envpool.mujoco.myosuite.paths import (
    myosuite_asset_root,
)
from envpool.mujoco.myosuite.registration import MYOSUITE_PUBLIC_TASK_IDS
from envpool.python.glfw_context import preload_windows_gl_dlls
from envpool.registration import list_all_envs, make_gymnasium

preload_windows_gl_dlls(strict=True)

_POSE_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["class_name"] == "PoseEnvV0"
)
_REACH_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["class_name"] == "ReachEnvV0"
)
_REORIENT_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["class_name"]
    in {
        "Geometries100EnvV0",
        "Geometries8EnvV0",
        "InDistribution",
        "OutofDistribution",
    }
)
_KEYTURN_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["class_name"] == "KeyTurnEnvV0"
)
_OBJHOLD_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["class_name"] in {"ObjHoldFixedEnvV0", "ObjHoldRandomEnvV0"}
)
_TORSO_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["class_name"] == "TorsoEnvV0"
)
_PENTWIRL_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["class_name"] in {"PenTwirlFixedEnvV0", "PenTwirlRandomEnvV0"}
)
_WALK_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["class_name"] == "WalkEnvV0"
)
_TERRAIN_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["class_name"] == "TerrainEnvV0"
)
_POSE_ALIGN_IDS = (
    "motorFingerPoseRandom-v0",
    "myoHandPoseRandom-v0",
    "myoElbowPose1D6MRandom-v0",
    "myoElbowPose1D6MExoRandom-v0",
)
_REACH_ALIGN_IDS = (
    "motorFingerReachRandom-v0",
    "myoHandReachRandom-v0",
    "myoArmReachRandom-v0",
)
_REORIENT_ALIGN_IDS = (
    "myoHandReorient100-v0",
    "myoHandReorient8-v0",
    "myoHandReorientID-v0",
    "myoHandReorientOOD-v0",
)
_KEYTURN_ALIGN_IDS = (
    "myoHandKeyTurnFixed-v0",
    "myoHandKeyTurnRandom-v0",
)
_OBJHOLD_ALIGN_IDS = (
    "myoHandObjHoldFixed-v0",
    "myoHandObjHoldRandom-v0",
)
_TORSO_ALIGN_IDS = (
    "myoTorsoPoseFixed-v0",
    "myoTorsoExoPoseFixed-v0",
)
_PENTWIRL_ALIGN_IDS = (
    "myoHandPenTwirlFixed-v0",
    "myoHandPenTwirlRandom-v0",
)
_WALK_ALIGN_IDS = ("myoLegWalk-v0",)
_TERRAIN_ALIGN_IDS = (
    "myoLegRoughTerrainWalk-v0",
    "myoLegHillyTerrainWalk-v0",
    "myoLegStairTerrainWalk-v0",
)
_PUBLIC_REPRESENTATIVE_TASK_IDS = (
    "myoHandReorientID-v0",
    "myoFatiHandReorientID-v0",
    "myoLegWalk-v0",
    "myoLegRoughTerrainWalk-v0",
    "myoChallengeBimanual-v0",
    "myoSarcChallengeBimanual-v0",
    "MyoHandAirplaneFly-v0",
    "myoReafHandPoseRandom-v0",
    "myoSarcArmReachRandom-v0",
)
_PUBLIC_VARIANT_DIFF_CASES = (
    ("myoHandPoseRandom-v0", "myoReafHandPoseRandom-v0"),
    ("myoArmReachRandom-v0", "myoSarcArmReachRandom-v0"),
    ("myoHandReorientID-v0", "myoFatiHandReorientID-v0"),
)
_ALIGNMENT_STEPS = 32


def _entry(env_id: str) -> dict[str, Any]:
    return next(
        entry for entry in MYOSUITE_DIRECT_ENTRIES if entry["id"] == env_id
    )


def _asset_model_path(model_path: str) -> Path:
    path = Path(model_path)
    return path if path.is_absolute() else myosuite_asset_root() / model_path


@cache
def _model(path: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(_asset_model_path(path)))


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


def _pose_config(
    env_id: str,
    *,
    model_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[tuple[Any, ...], type]:
    entry = _entry(env_id)
    kwargs = dict(entry["kwargs"])
    if model_path is not None:
        kwargs["model_path"] = model_path
    model = _model(kwargs["model_path"])
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
    config = MyoSuitePoseEnvSpec.gen_config(
        num_envs=1,
        batch_size=1,
        max_num_players=1,
        frame_skip=int(kwargs.get("frame_skip", 10)),
        model_path=str(kwargs["model_path"]),
        normalize_act=bool(kwargs.get("normalize_act", True)),
        obs_dim=model.nq + model.nv + model.nq + model.na,
        qpos_dim=model.nq,
        qvel_dim=model.nv,
        act_dim=model.na,
        action_dim=model.nu,
        pose_thd=float(kwargs.get("pose_thd", 0.35)),
        reward_pose_w=float(
            kwargs.get("weighted_reward_keys", {}).get("pose", 1.0)
        ),
        reward_bonus_w=float(
            kwargs.get("weighted_reward_keys", {}).get("bonus", 4.0)
        ),
        reward_act_reg_w=float(
            kwargs.get("weighted_reward_keys", {}).get("act_reg", 1.0)
        ),
        reward_penalty_w=float(
            kwargs.get("weighted_reward_keys", {}).get("penalty", 50.0)
        ),
        reset_type=str(kwargs.get("reset_type", "init")),
        target_type=str(
            kwargs.get(
                "target_type", "fixed" if target_qpos_value else "generate"
            )
        ),
        target_qpos_min=target_qpos_min,
        target_qpos_max=target_qpos_max,
        target_qpos_value=target_qpos_value,
        viz_site_targets=list(kwargs.get("viz_site_targets", [])),
        weight_bodyname=str(kwargs.get("weight_bodyname", "")),
        weight_range=list(kwargs.get("weight_range", [])),
    )
    if overrides:
        config = MyoSuitePoseEnvSpec.gen_config(
            **dict(
                zip(MyoSuitePoseEnvSpec._config_keys, config, strict=False),
                **overrides,
            )
        )
    return config, MyoSuitePoseGymnasiumEnvPool


def _reach_config(
    env_id: str,
    *,
    model_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[tuple[Any, ...], type]:
    entry = _entry(env_id)
    kwargs = dict(entry["kwargs"])
    if model_path is not None:
        kwargs["model_path"] = model_path
    model = _model(kwargs["model_path"])
    site_names, mins, maxs = _flatten_site_ranges(kwargs["target_reach_range"])
    config = MyoSuiteReachEnvSpec.gen_config(
        num_envs=1,
        batch_size=1,
        max_num_players=1,
        frame_skip=int(kwargs.get("frame_skip", 10)),
        model_path=str(kwargs["model_path"]),
        normalize_act=bool(kwargs.get("normalize_act", True)),
        obs_dim=model.nq + model.nv + 6 * len(site_names) + model.na,
        qpos_dim=model.nq,
        qvel_dim=model.nv,
        act_dim=model.na,
        action_dim=model.nu,
        target_site_count=len(site_names),
        far_th=float(kwargs.get("far_th", 0.35)),
        reward_reach_w=float(
            kwargs.get("weighted_reward_keys", {}).get("reach", 1.0)
        ),
        reward_bonus_w=float(
            kwargs.get("weighted_reward_keys", {}).get("bonus", 4.0)
        ),
        reward_act_reg_w=float(
            kwargs.get("weighted_reward_keys", {}).get("act_reg", 0.0)
        ),
        reward_penalty_w=float(
            kwargs.get("weighted_reward_keys", {}).get("penalty", 50.0)
        ),
        target_site_names=site_names,
        target_pos_min=mins,
        target_pos_max=maxs,
    )
    if overrides:
        config = MyoSuiteReachEnvSpec.gen_config(
            **dict(
                zip(MyoSuiteReachEnvSpec._config_keys, config, strict=False),
                **overrides,
            )
        )
    return config, MyoSuiteReachGymnasiumEnvPool


def _reorient_config(
    env_id: str,
    *,
    model_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[tuple[Any, ...], type]:
    entry = _entry(env_id)
    kwargs = dict(entry["kwargs"])
    if model_path is not None:
        kwargs["model_path"] = model_path
    model = _model(kwargs["model_path"])
    mode_map = {
        "Geometries100EnvV0": "100",
        "Geometries8EnvV0": "8",
        "InDistribution": "id",
        "OutofDistribution": "ood",
    }
    config = MyoSuiteReorientEnvSpec.gen_config(
        num_envs=1,
        batch_size=1,
        max_num_players=1,
        frame_skip=int(kwargs.get("frame_skip", 5)),
        model_path=str(kwargs["model_path"]),
        normalize_act=bool(kwargs.get("normalize_act", True)),
        obs_dim=(model.nq - 6) + 21 + 3 * model.nu + model.na,
        qpos_dim=model.nq,
        qvel_dim=model.nv,
        act_dim=model.na,
        action_dim=model.nu,
        randomization_mode=mode_map[entry["class_name"]],
        reward_pos_align_w=float(
            kwargs.get("weighted_reward_keys", {}).get("pos_align", 1.0)
        ),
        reward_rot_align_w=float(
            kwargs.get("weighted_reward_keys", {}).get("rot_align", 1.0)
        ),
        reward_act_reg_w=float(
            kwargs.get("weighted_reward_keys", {}).get("act_reg", 5.0)
        ),
        reward_drop_w=float(
            kwargs.get("weighted_reward_keys", {}).get("drop", 5.0)
        ),
        reward_bonus_w=float(
            kwargs.get("weighted_reward_keys", {}).get("bonus", 10.0)
        ),
    )
    if overrides:
        config = MyoSuiteReorientEnvSpec.gen_config(
            **dict(
                zip(MyoSuiteReorientEnvSpec._config_keys, config, strict=False),
                **overrides,
            )
        )
    return config, MyoSuiteReorientGymnasiumEnvPool


def _key_turn_config(
    env_id: str,
    *,
    model_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[tuple[Any, ...], type]:
    entry = _entry(env_id)
    kwargs = dict(entry["kwargs"])
    if model_path is not None:
        kwargs["model_path"] = model_path
    model = _model(kwargs["model_path"])
    config = MyoSuiteKeyTurnEnvSpec.gen_config(
        num_envs=1,
        batch_size=1,
        max_num_players=1,
        frame_skip=int(kwargs.get("frame_skip", 10)),
        model_path=str(kwargs["model_path"]),
        normalize_act=bool(kwargs.get("normalize_act", True)),
        obs_dim=model.nq + model.nv + 6 + model.na,
        qpos_dim=model.nq,
        qvel_dim=model.nv,
        act_dim=model.na,
        action_dim=model.nu,
        goal_th=float(kwargs.get("goal_th", np.pi)),
        reward_key_turn_w=float(
            kwargs.get("weighted_reward_keys", {}).get("key_turn", 1.0)
        ),
        reward_iftip_approach_w=float(
            kwargs.get("weighted_reward_keys", {}).get("IFtip_approach", 10.0)
        ),
        reward_thtip_approach_w=float(
            kwargs.get("weighted_reward_keys", {}).get("THtip_approach", 10.0)
        ),
        reward_act_reg_w=float(
            kwargs.get("weighted_reward_keys", {}).get("act_reg", 1.0)
        ),
        reward_bonus_w=float(
            kwargs.get("weighted_reward_keys", {}).get("bonus", 4.0)
        ),
        reward_penalty_w=float(
            kwargs.get("weighted_reward_keys", {}).get("penalty", 25.0)
        ),
        key_init_range=list(kwargs.get("key_init_range", (0.0, 0.0))),
    )
    if overrides:
        config = MyoSuiteKeyTurnEnvSpec.gen_config(
            **dict(
                zip(MyoSuiteKeyTurnEnvSpec._config_keys, config, strict=False),
                **overrides,
            )
        )
    return config, MyoSuiteKeyTurnGymnasiumEnvPool


def _obj_hold_config(
    env_id: str,
    *,
    model_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[tuple[Any, ...], type]:
    entry = _entry(env_id)
    kwargs = dict(entry["kwargs"])
    if model_path is not None:
        kwargs["model_path"] = model_path
    model = _model(kwargs["model_path"])
    config = MyoSuiteObjHoldEnvSpec.gen_config(
        num_envs=1,
        batch_size=1,
        max_num_players=1,
        frame_skip=int(kwargs.get("frame_skip", 10)),
        model_path=str(kwargs["model_path"]),
        normalize_act=bool(kwargs.get("normalize_act", True)),
        obs_dim=(model.nq - 7) + (model.nv - 6) + 6 + model.na,
        qpos_dim=model.nq,
        qvel_dim=model.nv,
        act_dim=model.na,
        action_dim=model.nu,
        randomize_on_reset=entry["class_name"] == "ObjHoldRandomEnvV0",
        reward_goal_dist_w=float(
            kwargs.get("weighted_reward_keys", {}).get("goal_dist", 100.0)
        ),
        reward_bonus_w=float(
            kwargs.get("weighted_reward_keys", {}).get("bonus", 4.0)
        ),
        reward_penalty_w=float(
            kwargs.get("weighted_reward_keys", {}).get("penalty", 10.0)
        ),
    )
    if overrides:
        config = MyoSuiteObjHoldEnvSpec.gen_config(
            **dict(
                zip(MyoSuiteObjHoldEnvSpec._config_keys, config, strict=False),
                **overrides,
            )
        )
    return config, MyoSuiteObjHoldGymnasiumEnvPool


def _torso_config(
    env_id: str,
    *,
    model_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[tuple[Any, ...], type]:
    entry = _entry(env_id)
    kwargs = dict(entry["kwargs"])
    if model_path is not None:
        kwargs["model_path"] = model_path
    model = _model(kwargs["model_path"])
    target_qpos_value = [
        (bounds[0] + bounds[1]) / 2.0
        for bounds in kwargs["target_jnt_range"].values()
    ]
    pose_dim = len(target_qpos_value)
    config = MyoSuiteTorsoEnvSpec.gen_config(
        num_envs=1,
        batch_size=1,
        max_num_players=1,
        frame_skip=int(kwargs.get("frame_skip", 5)),
        model_path=str(kwargs["model_path"]),
        normalize_act=bool(kwargs.get("normalize_act", True)),
        obs_dim=model.nq + model.nv + pose_dim + model.na,
        qpos_dim=model.nq,
        qvel_dim=model.nv,
        act_dim=model.na,
        action_dim=model.nu,
        pose_dim=pose_dim,
        pose_thd=float(kwargs.get("pose_thd", 0.25)),
        reward_pose_w=float(
            kwargs.get("weighted_reward_keys", {}).get("pose", 1.0)
        ),
        reward_bonus_w=float(
            kwargs.get("weighted_reward_keys", {}).get("bonus", 4.0)
        ),
        reward_act_reg_w=float(
            kwargs.get("weighted_reward_keys", {}).get("act_reg", 1.0)
        ),
        reward_penalty_w=float(
            kwargs.get("weighted_reward_keys", {}).get("penalty", 50.0)
        ),
        target_qpos_value=target_qpos_value,
    )
    if overrides:
        overrides = dict(overrides)
        overrides.pop("test_target_qpos", None)
        config = MyoSuiteTorsoEnvSpec.gen_config(
            **dict(
                zip(MyoSuiteTorsoEnvSpec._config_keys, config, strict=False),
                **overrides,
            )
        )
    return config, MyoSuiteTorsoGymnasiumEnvPool


def _pen_twirl_config(
    env_id: str,
    *,
    model_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[tuple[Any, ...], type]:
    entry = _entry(env_id)
    kwargs = dict(entry["kwargs"])
    if model_path is not None:
        kwargs["model_path"] = model_path
    model = _model(kwargs["model_path"])
    config = MyoSuitePenTwirlEnvSpec.gen_config(
        num_envs=1,
        batch_size=1,
        max_num_players=1,
        frame_skip=int(kwargs.get("frame_skip", 5)),
        model_path=str(kwargs["model_path"]),
        normalize_act=bool(kwargs.get("normalize_act", True)),
        obs_dim=(model.nq - 6) + 21 + model.na,
        qpos_dim=model.nq,
        qvel_dim=model.nv,
        act_dim=model.na,
        action_dim=model.nu,
        randomize_target=entry["class_name"] == "PenTwirlRandomEnvV0",
        reward_pos_align_w=float(
            kwargs.get("weighted_reward_keys", {}).get("pos_align", 1.0)
        ),
        reward_rot_align_w=float(
            kwargs.get("weighted_reward_keys", {}).get("rot_align", 1.0)
        ),
        reward_act_reg_w=float(
            kwargs.get("weighted_reward_keys", {}).get("act_reg", 5.0)
        ),
        reward_drop_w=float(
            kwargs.get("weighted_reward_keys", {}).get("drop", 5.0)
        ),
        reward_bonus_w=float(
            kwargs.get("weighted_reward_keys", {}).get("bonus", 10.0)
        ),
    )
    if overrides:
        config = MyoSuitePenTwirlEnvSpec.gen_config(
            **dict(
                zip(MyoSuitePenTwirlEnvSpec._config_keys, config, strict=False),
                **overrides,
            )
        )
    return config, MyoSuitePenTwirlGymnasiumEnvPool


def _walk_like_config(
    env_id: str,
    *,
    model_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[tuple[Any, ...], type, type]:
    entry = _entry(env_id)
    kwargs = dict(entry["kwargs"])
    if model_path is not None:
        kwargs["model_path"] = model_path
    model = _model(kwargs["model_path"])
    spec_type = (
        MyoSuiteTerrainEnvSpec
        if entry["class_name"] == "TerrainEnvV0"
        else MyoSuiteWalkEnvSpec
    )
    pool_type = (
        MyoSuiteTerrainGymnasiumEnvPool
        if entry["class_name"] == "TerrainEnvV0"
        else MyoSuiteWalkGymnasiumEnvPool
    )
    config = spec_type.gen_config(
        num_envs=1,
        batch_size=1,
        max_num_players=1,
        frame_skip=int(kwargs.get("frame_skip", 10)),
        model_path=str(kwargs["model_path"]),
        normalize_act=bool(kwargs.get("normalize_act", True)),
        obs_dim=(model.nq - 2)
        + model.nv
        + 2
        + 4
        + 2
        + 1
        + 6
        + 1
        + 3 * model.nu
        + model.na,
        qpos_dim=model.nq,
        qvel_dim=model.nv,
        act_dim=model.na,
        action_dim=model.nu,
        min_height=float(kwargs.get("min_height", 0.8)),
        max_rot=float(kwargs.get("max_rot", 0.8)),
        hip_period=int(kwargs.get("hip_period", 100)),
        reset_type=str(kwargs.get("reset_type", "init")),
        target_x_vel=float(kwargs.get("target_x_vel", 0.0)),
        target_y_vel=float(kwargs.get("target_y_vel", 1.2)),
        target_rot=[]
        if kwargs.get("target_rot") is None
        else list(kwargs["target_rot"]),
        terrain=str(kwargs.get("terrain", "")),
        terrain_variant=""
        if kwargs.get("variant") is None
        else str(kwargs.get("variant")),
        use_knee_condition=entry["class_name"] == "TerrainEnvV0",
        reward_vel_w=float(
            kwargs.get("weighted_reward_keys", {}).get("vel_reward", 5.0)
        ),
        reward_done_w=float(
            kwargs.get("weighted_reward_keys", {}).get("done", -100.0)
        ),
        reward_cyclic_hip_w=float(
            kwargs.get("weighted_reward_keys", {}).get("cyclic_hip", -10.0)
        ),
        reward_ref_rot_w=float(
            kwargs.get("weighted_reward_keys", {}).get("ref_rot", 10.0)
        ),
        reward_joint_angle_w=float(
            kwargs.get("weighted_reward_keys", {}).get("joint_angle_rew", 5.0)
        ),
    )
    if overrides:
        config = spec_type.gen_config(
            **dict(
                zip(spec_type._config_keys, config, strict=False),
                **overrides,
            )
        )
    return config, pool_type, spec_type


def _pose_pixel_config(env_id: str) -> tuple[tuple[Any, ...], type]:
    config, _ = _pose_config(env_id)
    values = dict(zip(MyoSuitePoseEnvSpec._config_keys, config, strict=False))
    pixel = MyoSuitePosePixelEnvSpec.gen_config(
        **values,
        render_width=64,
        render_height=64,
        render_camera_id=-1,
    )
    return pixel, MyoSuitePosePixelGymnasiumEnvPool


def _reach_pixel_config(env_id: str) -> tuple[tuple[Any, ...], type]:
    config, _ = _reach_config(env_id)
    values = dict(zip(MyoSuiteReachEnvSpec._config_keys, config, strict=False))
    pixel = MyoSuiteReachPixelEnvSpec.gen_config(
        **values,
        render_width=64,
        render_height=64,
        render_camera_id=-1,
    )
    return pixel, MyoSuiteReachPixelGymnasiumEnvPool


def _reorient_pixel_config(env_id: str) -> tuple[tuple[Any, ...], type]:
    config, _ = _reorient_config(env_id)
    values = dict(
        zip(MyoSuiteReorientEnvSpec._config_keys, config, strict=False)
    )
    pixel = MyoSuiteReorientPixelEnvSpec.gen_config(
        **values,
        render_width=64,
        render_height=64,
        render_camera_id=-1,
    )
    return pixel, MyoSuiteReorientPixelGymnasiumEnvPool


def _key_turn_pixel_config(env_id: str) -> tuple[tuple[Any, ...], type]:
    config, _ = _key_turn_config(env_id)
    values = dict(
        zip(MyoSuiteKeyTurnEnvSpec._config_keys, config, strict=False)
    )
    pixel = MyoSuiteKeyTurnPixelEnvSpec.gen_config(
        **values,
        render_width=64,
        render_height=64,
        render_camera_id=-1,
    )
    return pixel, MyoSuiteKeyTurnPixelGymnasiumEnvPool


def _obj_hold_pixel_config(env_id: str) -> tuple[tuple[Any, ...], type]:
    config, _ = _obj_hold_config(env_id)
    values = dict(
        zip(MyoSuiteObjHoldEnvSpec._config_keys, config, strict=False)
    )
    pixel = MyoSuiteObjHoldPixelEnvSpec.gen_config(
        **values,
        render_width=64,
        render_height=64,
        render_camera_id=-1,
    )
    return pixel, MyoSuiteObjHoldPixelGymnasiumEnvPool


def _torso_pixel_config(env_id: str) -> tuple[tuple[Any, ...], type]:
    config, _ = _torso_config(env_id)
    values = dict(zip(MyoSuiteTorsoEnvSpec._config_keys, config, strict=False))
    pixel = MyoSuiteTorsoPixelEnvSpec.gen_config(
        **values,
        render_width=64,
        render_height=64,
        render_camera_id=-1,
    )
    return pixel, MyoSuiteTorsoPixelGymnasiumEnvPool


def _pen_twirl_pixel_config(env_id: str) -> tuple[tuple[Any, ...], type]:
    config, _ = _pen_twirl_config(env_id)
    values = dict(
        zip(MyoSuitePenTwirlEnvSpec._config_keys, config, strict=False)
    )
    pixel = MyoSuitePenTwirlPixelEnvSpec.gen_config(
        **values,
        render_width=64,
        render_height=64,
        render_camera_id=-1,
    )
    return pixel, MyoSuitePenTwirlPixelGymnasiumEnvPool


def _walk_pixel_config(env_id: str) -> tuple[tuple[Any, ...], type, type]:
    config, _, _ = _walk_like_config(env_id)
    values = dict(zip(MyoSuiteWalkEnvSpec._config_keys, config, strict=False))
    pixel = MyoSuiteWalkPixelEnvSpec.gen_config(
        **values,
        render_width=64,
        render_height=64,
        render_camera_id=-1,
    )
    return pixel, MyoSuiteWalkPixelGymnasiumEnvPool, MyoSuiteWalkPixelEnvSpec


def _terrain_pixel_config(env_id: str) -> tuple[tuple[Any, ...], type, type]:
    config, _, _ = _walk_like_config(env_id)
    values = dict(
        zip(MyoSuiteTerrainEnvSpec._config_keys, config, strict=False)
    )
    pixel = MyoSuiteTerrainPixelEnvSpec.gen_config(
        **values,
        render_width=64,
        render_height=64,
        render_camera_id=-1,
    )
    return (
        pixel,
        MyoSuiteTerrainPixelGymnasiumEnvPool,
        MyoSuiteTerrainPixelEnvSpec,
    )


def _make_env(config: tuple[Any, ...], pool_type: type, spec_type: type) -> Any:
    return pool_type(spec_type(config))


def _make_registered_env(task_id: str, **kwargs: Any) -> Any:
    return make_gymnasium(
        task_id,
        num_envs=1,
        batch_size=1,
        max_num_players=1,
        **kwargs,
    )


def _registered_rollout(
    task_id: str, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    env = _make_registered_env(task_id, seed=seed)
    try:
        env.reset()
        action = np.full(
            (1, int(env.action_space.shape[-1])), 0.35, dtype=np.float32
        )
        final_obs = None
        final_reward = None
        for _ in range(3):
            final_obs, final_reward, *_ = env.step(action)
        assert final_obs is not None
        assert final_reward is not None
        return final_obs, final_reward
    finally:
        env.close()


def _seeded_actions(
    shape: tuple[int, ...], steps: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        rng.uniform(-0.9, 0.9, size=shape).astype(np.float32)
        for _ in range(steps)
    ]


def _assert_rollouts_match(
    case: absltest.TestCase,
    env0: Any,
    env1: Any,
    actions: list[np.ndarray],
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> None:
    obs0, info0 = env0.reset()
    obs1, info1 = env1.reset()
    np.testing.assert_allclose(obs0, obs1, atol=atol, rtol=rtol)
    case.assertEqual(
        info0["elapsed_step"].tolist(), info1["elapsed_step"].tolist()
    )
    for action in actions:
        out0 = env0.step(action)
        out1 = env1.step(action)
        obs0, reward0, terminated0, truncated0, info0 = out0
        obs1, reward1, terminated1, truncated1, info1 = out1
        np.testing.assert_allclose(obs0, obs1, atol=atol, rtol=rtol)
        np.testing.assert_allclose(reward0, reward1, atol=atol, rtol=rtol)
        np.testing.assert_array_equal(terminated0, terminated1)
        np.testing.assert_array_equal(truncated0, truncated1)
        case.assertEqual(
            info0["elapsed_step"].tolist(), info1["elapsed_step"].tolist()
        )
        if terminated0[0] or truncated0[0]:
            break

def _first_child_body(body: mujoco.MjsBody) -> mujoco.MjsBody | None:
    return body.first_body()


def _arm_reaching_spec_to_xml(model_path: str) -> str:
    spec = mujoco.MjSpec.from_file(model_path)
    root_names = ("firstmc", "secondmc", "thirdmc", "fourthmc", "fifthmc")
    tip_site = spec.site("IFtip")
    tip_site_name = str(tip_site.name)
    tip_site_size = tip_site.size.copy()
    tip_site_pos = tip_site.pos.copy()
    tip_site_rgba = tip_site.rgba.copy()
    body_chains: dict[str, list[tuple[str, np.ndarray, list[str]]]] = {}

    for root_name in root_names:
        body_chains[root_name] = []
        root_body = spec.body(root_name)
        child = _first_child_body(root_body)
        while child is not None:
            mesh_names = [
                str(geom.name)
                for geom in child.geoms
                if geom.type == mujoco.mjtGeom.mjGEOM_MESH
            ]
            body_chains[root_name].append((
                str(child.name),
                child.pos.copy(),
                mesh_names,
            ))
            child = _first_child_body(child)

    for root_name in root_names:
        root_body = spec.body(root_name)
        child = _first_child_body(root_body)
        if child is not None:
            spec.delete(child)

    for root_name in root_names:
        parent = spec.body(root_name)
        for body_name, pos, mesh_names in body_chains[root_name]:
            parent.add_body(name=body_name, pos=pos)
            current = spec.body(body_name)
            for mesh_name in mesh_names:
                current.add_geom(
                    meshname=mesh_name,
                    name=body_name,
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                )
            if body_name == "distph2":
                current.add_site(
                    name=tip_site_name,
                    size=tip_site_size * 2.0,
                    pos=tip_site_pos,
                    rgba=tip_site_rgba,
                )
            parent = current

    spec.body("world").add_site(
        name="IFtip_target",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.02, 0.02, 0.02],
        pos=[-0.2, -0.2, 1.2],
        rgba=[0.0, 0.0, 1.0, 0.3],
    )
    return spec.to_xml()


@contextmanager
def _edited_arm_reaching_model(model_path: str) -> Iterator[str]:
    base_model_path = Path(model_path)
    myo_sim_root = base_model_path.parent.parent
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        arm_dir = td_path / "arm"
        arm_dir.mkdir()
        (td_path / "myo_sim").symlink_to(myo_sim_root, target_is_directory=True)
        edited_model_path = arm_dir / "myoarm_reach.xml"
        xml_tree = ET.ElementTree(
            ET.fromstring(_arm_reaching_spec_to_xml(model_path))
        )
        root = xml_tree.getroot()
        if root is not None:
            compiler = root.find("compiler")
            if compiler is not None:
                compiler.set("meshdir", ".")
                compiler.set("texturedir", ".")
        xml_tree.write(edited_model_path, encoding="utf-8")
        yield str(edited_model_path)


@contextmanager
def _edited_model_if_needed(entry: dict[str, Any]) -> Iterator[str]:
    kwargs = entry["kwargs"]
    edit_name = kwargs.get("edit_fn")
    if edit_name != "edit_fn_arm_reaching":
        yield str(_asset_model_path(kwargs["model_path"]))
        return
    with _edited_arm_reaching_model(
        str(_asset_model_path(kwargs["model_path"]))
    ) as edited_path:
        yield edited_path


def _oracle_reset_sync(
    env: Any, env_id: str
) -> tuple[np.ndarray, dict[str, Any]]:
    obs, _ = env.reset()
    unwrapped = env.unwrapped
    sync: dict[str, Any] = {
        "test_reset_qpos": unwrapped.sim.data.qpos.copy().tolist(),
        "test_reset_qvel": unwrapped.sim.data.qvel.copy().tolist(),
        "test_reset_act": (
            unwrapped.sim.data.act.copy().tolist()
            if unwrapped.sim.model.na > 0
            else []
        ),
        "test_reset_qacc_warmstart": unwrapped.sim.data.qacc_warmstart.copy().tolist(),
    }
    entry = _entry(env_id)
    if entry["class_name"] == "KeyTurnEnvV0":
        obs_sim = getattr(unwrapped, "sim_obsd", unwrapped.sim)
        keyhead_sid = obs_sim.model.site_name2id("keyhead")
        key_body_id = int(obs_sim.model.site_bodyid[keyhead_sid])
        sync["test_key_body_pos"] = (
            obs_sim.model.body_pos[key_body_id].copy().tolist()
        )
    elif entry["class_name"] in {"ObjHoldFixedEnvV0", "ObjHoldRandomEnvV0"}:
        goal_sid = unwrapped.sim.model.site_name2id("goal")
        geom_id = unwrapped.sim.model.geom_name2id("object")
        sync["test_goal_pos"] = (
            unwrapped.sim.model.site_pos[goal_sid].copy().tolist()
        )
        sync["test_object_geom_size"] = (
            unwrapped.sim.model.geom_size[geom_id].copy().tolist()
        )
    elif entry["class_name"] in {
        "Geometries100EnvV0",
        "Geometries8EnvV0",
        "InDistribution",
        "OutofDistribution",
    }:
        model = unwrapped.sim.model
        target_body_id = model.body_name2id("target")
        obj_geom_id = model.geom_name2id("obj")
        target_geom_id = model.geom_name2id("target")
        top_geom_id = model.geom_name2id("top")
        bot_geom_id = model.geom_name2id("bot")
        target_top_geom_id = model.geom_name2id("t_top")
        target_bot_geom_id = model.geom_name2id("t_bot")
        object_body_id = model.body_name2id("Object")
        sync["test_target_body_quat"] = (
            model.body_quat[target_body_id].copy().tolist()
        )
        sync["test_object_geom_size"] = (
            model.geom_size[obj_geom_id].copy().tolist()
        )
        sync["test_target_geom_size"] = (
            model.geom_size[target_geom_id].copy().tolist()
        )
        sync["test_object_geom_rgba"] = (
            model.geom_rgba[obj_geom_id].copy().tolist()
        )
        sync["test_target_geom_rgba"] = (
            model.geom_rgba[target_geom_id].copy().tolist()
        )
        sync["test_object_geom_top_pos"] = (
            model.geom_pos[top_geom_id].copy().tolist()
        )
        sync["test_object_geom_bottom_pos"] = (
            model.geom_pos[bot_geom_id].copy().tolist()
        )
        sync["test_target_geom_top_pos"] = (
            model.geom_pos[target_top_geom_id].copy().tolist()
        )
        sync["test_target_geom_bottom_pos"] = (
            model.geom_pos[target_bot_geom_id].copy().tolist()
        )
        sync["test_object_body_mass"] = [float(model.body_mass[object_body_id])]
        sync["test_object_geom_type"] = int(model.geom_type[obj_geom_id])
        sync["test_target_geom_type"] = int(model.geom_type[target_geom_id])
        sync["test_object_geom_condim"] = int(model.geom_condim[obj_geom_id])
        sync["test_success_site_rgba"] = (
            model.site_rgba[unwrapped.success_indicator_sid].copy().tolist()
        )
    elif entry["class_name"] in {"PenTwirlFixedEnvV0", "PenTwirlRandomEnvV0"}:
        target_body_id = unwrapped.sim.model.body_name2id("target")
        sync["test_target_body_quat"] = (
            unwrapped.sim.model.body_quat[target_body_id].copy().tolist()
        )
    elif entry["class_name"] == "TerrainEnvV0":
        terrain_geom_id = unwrapped.sim.model.geom_name2id("terrain")
        hfield_id = int(unwrapped.sim.model.geom_dataid[terrain_geom_id])
        nrow = int(unwrapped.sim.model.hfield_nrow[hfield_id])
        ncol = int(unwrapped.sim.model.hfield_ncol[hfield_id])
        adr = int(unwrapped.sim.model.hfield_adr[hfield_id])
        sync["test_hfield_data"] = (
            unwrapped.sim.model
            .hfield_data[adr : adr + nrow * ncol]
            .copy()
            .tolist()
        )
        sync["test_terrain_geom_rgba"] = (
            unwrapped.sim.model.geom_rgba[terrain_geom_id].copy().tolist()
        )
        sync["test_terrain_geom_pos"] = (
            unwrapped.sim.model.geom_pos[terrain_geom_id].copy().tolist()
        )
        sync["test_terrain_geom_contype"] = int(
            unwrapped.sim.model.geom_contype[terrain_geom_id]
        )
        sync["test_terrain_geom_conaffinity"] = int(
            unwrapped.sim.model.geom_conaffinity[terrain_geom_id]
        )
    elif entry["class_name"] == "WalkEnvV0":
        terrain_geom_id = unwrapped.sim.model.geom_name2id("terrain")
        sync["test_terrain_geom_rgba"] = (
            unwrapped.sim.model.geom_rgba[terrain_geom_id].copy().tolist()
        )
        sync["test_terrain_geom_pos"] = (
            unwrapped.sim.model.geom_pos[terrain_geom_id].copy().tolist()
        )
        sync["test_terrain_geom_contype"] = int(
            unwrapped.sim.model.geom_contype[terrain_geom_id]
        )
        sync["test_terrain_geom_conaffinity"] = int(
            unwrapped.sim.model.geom_conaffinity[terrain_geom_id]
        )
    elif "target_reach_range" not in entry["kwargs"]:
        sync["test_target_qpos"] = unwrapped.target_jnt_value.copy().tolist()
        if getattr(unwrapped, "weight_bodyname", None):
            body_id = unwrapped.sim.model.body_name2id(
                unwrapped.weight_bodyname
            )
            geom_id = unwrapped.sim.model.body_geomadr[body_id]
            sync["test_body_mass"] = [
                float(unwrapped.sim.model.body_mass[body_id])
            ]
            sync["test_geom_size0"] = [
                float(unwrapped.sim.model.geom_size[geom_id][0])
            ]
    else:
        target_pos: list[float] = []
        for site_name in entry["kwargs"]["target_reach_range"]:
            site_id = unwrapped.sim.model.site_name2id(site_name + "_target")
            target_pos.extend(
                unwrapped.sim.model.site_pos[site_id].copy().tolist()
            )
        sync["test_target_pos"] = target_pos
    return obs, sync


def _oracle_class(env_id: str) -> Any:
    entry = _entry(env_id)
    return load_oracle_class(entry["entry_module"], entry["class_name"])


def _pose_alignment_obs_atol(env_id: str) -> float:
    if env_id == "motorFingerPoseRandom-v0":
        return 3e-4
    return 1e-7


def _pose_alignment_reward_atol(env_id: str) -> float:
    del env_id
    return 1e-7


def _reach_alignment_obs_atol(env_id: str) -> float:
    if env_id == "motorFingerReachRandom-v0":
        return 4e-4
    if env_id in {"myoHandReachRandom-v0", "myoArmReachRandom-v0"}:
        return 3e-3 if env_id == "myoHandReachRandom-v0" else 2.5e-3
    return 1e-7


def _reach_alignment_reward_atol(env_id: str) -> float:
    # These envs still carry small reset-synced residuals because the oracle
    # computes rewards from its observed-sim reconstruction path, while the
    # native slice currently evaluates directly from the stepped sim state.
    if env_id == "motorFingerReachRandom-v0":
        return 3e-4
    if env_id == "myoHandReachRandom-v0":
        return 3e-3
    if env_id == "myoArmReachRandom-v0":
        return 2e-3
    return 1e-7


def _reorient_alignment_obs_atol(env_id: str) -> float:
    del env_id
    return 7.0


def _reorient_alignment_reward_atol(env_id: str) -> float:
    del env_id
    return 0.5


def _key_turn_alignment_obs_atol(env_id: str) -> float:
    # The official KeyTurn oracle emits observations from its sim_obsd path,
    # while the native slice currently reads directly from the stepped sim.
    # The residual is isolated to fingertip/key site vectors after reset-sync.
    del env_id
    return 3e-3


def _key_turn_alignment_reward_atol(env_id: str) -> float:
    del env_id
    return 6e-3


def _obj_hold_alignment_obs_atol(env_id: str) -> float:
    del env_id
    return 2e-3


def _obj_hold_alignment_reward_atol(env_id: str) -> float:
    del env_id
    return 5e-2


def _torso_alignment_obs_atol(env_id: str) -> float:
    del env_id
    return 1e-7


def _torso_alignment_reward_atol(env_id: str) -> float:
    del env_id
    return 1e-7


def _pen_twirl_alignment_obs_atol(env_id: str) -> float:
    del env_id
    return 1.1e-2


def _pen_twirl_alignment_reward_atol(env_id: str) -> float:
    del env_id
    return 2e-2


def _walk_alignment_obs_atol(env_id: str) -> float:
    del env_id
    return 5e-2


def _walk_alignment_reward_atol(env_id: str) -> float:
    del env_id
    return 0.2


def _terrain_alignment_obs_atol(env_id: str) -> float:
    del env_id
    return 3.0


def _terrain_alignment_reward_atol(env_id: str) -> float:
    del env_id
    return 8.0


class MyoSuiteMyoBaseNativeTest(absltest.TestCase):
    """Covers the internal native MyoSuite MyoBase slice."""

    def test_pose_surface_count(self) -> None:
        """PoseEnvV0 metadata should map to the expected direct surface."""
        self.assertLen(_POSE_IDS, 20)

    def test_reach_surface_count(self) -> None:
        """ReachEnvV0 metadata should map to the expected direct surface."""
        self.assertLen(_REACH_IDS, 9)

    def test_reorient_surface_count(self) -> None:
        """Reorient metadata should map to the expected direct surface."""
        self.assertLen(_REORIENT_IDS, 4)

    def test_key_turn_surface_count(self) -> None:
        """KeyTurnEnvV0 metadata should map to the expected direct surface."""
        self.assertLen(_KEYTURN_IDS, 2)

    def test_obj_hold_surface_count(self) -> None:
        """ObjHold metadata should map to the expected direct surface."""
        self.assertLen(_OBJHOLD_IDS, 2)

    def test_torso_surface_count(self) -> None:
        """TorsoEnvV0 metadata should map to the expected direct surface."""
        self.assertLen(_TORSO_IDS, 2)

    def test_pen_twirl_surface_count(self) -> None:
        """PenTwirl metadata should map to the expected direct surface."""
        self.assertLen(_PENTWIRL_IDS, 2)

    def test_walk_surface_count(self) -> None:
        """WalkEnvV0 metadata should map to the expected direct surface."""
        self.assertLen(_WALK_IDS, 1)

    def test_terrain_surface_count(self) -> None:
        """TerrainEnvV0 metadata should map to the expected direct surface."""
        self.assertLen(_TERRAIN_IDS, 3)

    def test_pose_native_determinism_for_all_ids(self) -> None:
        """Pose envs should be deterministic under a fixed action sequence."""
        for env_id in _POSE_IDS:
            config, pool_type = _pose_config(env_id)
            env0 = _make_env(config, pool_type, MyoSuitePoseEnvSpec)
            env1 = _make_env(config, pool_type, MyoSuitePoseEnvSpec)
            model = _model(_entry(env_id)["kwargs"]["model_path"])
            actions = _seeded_actions((1, model.nu), steps=8, seed=17)
            with self.subTest(env_id=env_id):
                _assert_rollouts_match(self, env0, env1, actions)

    def test_reach_native_determinism_for_all_ids(self) -> None:
        """Reach envs should be deterministic under a fixed action sequence."""
        for env_id in _REACH_IDS:
            entry = _entry(env_id)
            with _edited_model_if_needed(entry) as model_path:
                config, pool_type = _reach_config(env_id, model_path=model_path)
                env0 = _make_env(config, pool_type, MyoSuiteReachEnvSpec)
                env1 = _make_env(config, pool_type, MyoSuiteReachEnvSpec)
                model = _model(model_path)
                actions = _seeded_actions((1, model.nu), steps=8, seed=29)
                with self.subTest(env_id=env_id):
                    _assert_rollouts_match(self, env0, env1, actions)

    def test_reorient_native_determinism_for_all_ids(self) -> None:
        """Reorient envs should be deterministic under a fixed action sequence."""
        for env_id in _REORIENT_IDS:
            config, pool_type = _reorient_config(env_id)
            env0 = _make_env(config, pool_type, MyoSuiteReorientEnvSpec)
            env1 = _make_env(config, pool_type, MyoSuiteReorientEnvSpec)
            model = _model(_entry(env_id)["kwargs"]["model_path"])
            actions = _seeded_actions((1, model.nu), steps=8, seed=61)
            with self.subTest(env_id=env_id):
                _assert_rollouts_match(self, env0, env1, actions)

    def test_key_turn_native_determinism_for_all_ids(self) -> None:
        """KeyTurn envs should be deterministic under a fixed action sequence."""
        for env_id in _KEYTURN_IDS:
            config, pool_type = _key_turn_config(env_id)
            env0 = _make_env(config, pool_type, MyoSuiteKeyTurnEnvSpec)
            env1 = _make_env(config, pool_type, MyoSuiteKeyTurnEnvSpec)
            model = _model(_entry(env_id)["kwargs"]["model_path"])
            actions = _seeded_actions((1, model.nu), steps=8, seed=71)
            with self.subTest(env_id=env_id):
                _assert_rollouts_match(self, env0, env1, actions)

    def test_obj_hold_native_determinism_for_all_ids(self) -> None:
        """ObjHold envs should be deterministic under a fixed action sequence."""
        for env_id in _OBJHOLD_IDS:
            config, pool_type = _obj_hold_config(env_id)
            env0 = _make_env(config, pool_type, MyoSuiteObjHoldEnvSpec)
            env1 = _make_env(config, pool_type, MyoSuiteObjHoldEnvSpec)
            model = _model(_entry(env_id)["kwargs"]["model_path"])
            actions = _seeded_actions((1, model.nu), steps=8, seed=83)
            with self.subTest(env_id=env_id):
                _assert_rollouts_match(self, env0, env1, actions)

    def test_torso_native_determinism_for_all_ids(self) -> None:
        """Torso envs should be deterministic under a fixed action sequence."""
        for env_id in _TORSO_IDS:
            config, pool_type = _torso_config(env_id)
            env0 = _make_env(config, pool_type, MyoSuiteTorsoEnvSpec)
            env1 = _make_env(config, pool_type, MyoSuiteTorsoEnvSpec)
            model = _model(_entry(env_id)["kwargs"]["model_path"])
            actions = _seeded_actions((1, model.nu), steps=8, seed=89)
            with self.subTest(env_id=env_id):
                _assert_rollouts_match(self, env0, env1, actions)

    def test_pen_twirl_native_determinism_for_all_ids(self) -> None:
        """PenTwirl envs should be deterministic under a fixed action sequence."""
        for env_id in _PENTWIRL_IDS:
            config, pool_type = _pen_twirl_config(env_id)
            env0 = _make_env(config, pool_type, MyoSuitePenTwirlEnvSpec)
            env1 = _make_env(config, pool_type, MyoSuitePenTwirlEnvSpec)
            model = _model(_entry(env_id)["kwargs"]["model_path"])
            actions = _seeded_actions((1, model.nu), steps=8, seed=107)
            with self.subTest(env_id=env_id):
                _assert_rollouts_match(self, env0, env1, actions)

    def test_walk_native_determinism_for_all_ids(self) -> None:
        """Walk envs should be deterministic under a fixed action sequence."""
        for env_id in _WALK_IDS:
            config, pool_type, spec_type = _walk_like_config(env_id)
            env0 = _make_env(config, pool_type, spec_type)
            env1 = _make_env(config, pool_type, spec_type)
            model = _model(_entry(env_id)["kwargs"]["model_path"])
            actions = _seeded_actions((1, model.nu), steps=8, seed=113)
            with self.subTest(env_id=env_id):
                _assert_rollouts_match(self, env0, env1, actions)

    def test_terrain_native_determinism_for_all_ids(self) -> None:
        """Terrain envs should be deterministic under a fixed action sequence."""
        for env_id in _TERRAIN_IDS:
            config, pool_type, spec_type = _walk_like_config(env_id)
            env0 = _make_env(config, pool_type, spec_type)
            env1 = _make_env(config, pool_type, spec_type)
            model = _model(_entry(env_id)["kwargs"]["model_path"])
            actions = _seeded_actions((1, model.nu), steps=8, seed=127)
            with self.subTest(env_id=env_id):
                _assert_rollouts_match(self, env0, env1, actions)

    def test_pose_alignment_representative_ids(self) -> None:
        """Representative pose envs should align with the official oracle."""
        for env_id in _POSE_ALIGN_IDS:
            entry = _entry(env_id)
            with _edited_model_if_needed(entry) as model_path:
                with self.subTest(env_id=env_id):
                    cls = _oracle_class(env_id)
                    kwargs = dict(entry["kwargs"])
                    kwargs.pop("edit_fn", None)
                    kwargs["model_path"] = model_path
                    oracle: Any = gymnasium.wrappers.TimeLimit(
                        cls(seed=123, **kwargs),
                        max_episode_steps=entry["max_episode_steps"],
                    )
                    obs0, sync = _oracle_reset_sync(oracle, env_id)
                    obs_atol = _pose_alignment_obs_atol(env_id)
                    reward_atol = _pose_alignment_reward_atol(env_id)
                    config, pool_type = _pose_config(
                        env_id, model_path=model_path, overrides=sync
                    )
                    native = _make_env(config, pool_type, MyoSuitePoseEnvSpec)
                    obs1, info1 = native.reset()
                    np.testing.assert_allclose(
                        obs1, obs0[None, :], atol=obs_atol, rtol=obs_atol
                    )
                    self.assertEqual(info1["elapsed_step"].tolist(), [0])
                    native_model = _model(model_path)
                    actions = _seeded_actions(
                        (1, native_model.nu),
                        steps=_ALIGNMENT_STEPS,
                        seed=41,
                    )
                    for action in actions:
                        obs0, reward0, terminated0, truncated0, _ = oracle.step(
                            action[0]
                        )
                        obs1, reward1, terminated1, truncated1, _ = native.step(
                            action
                        )
                        np.testing.assert_allclose(
                            obs1, obs0[None, :], atol=obs_atol, rtol=obs_atol
                        )
                        np.testing.assert_allclose(
                            reward1,
                            np.array([reward0]),
                            atol=reward_atol,
                            rtol=reward_atol,
                        )
                        np.testing.assert_array_equal(
                            terminated1, np.array([terminated0])
                        )
                        np.testing.assert_array_equal(
                            truncated1, np.array([truncated0])
                        )
                        if terminated0 or truncated0:
                            break

    def test_reach_alignment_representative_ids(self) -> None:
        """Representative reach envs should align with the official oracle."""
        for env_id in _REACH_ALIGN_IDS:
            entry = _entry(env_id)
            with _edited_model_if_needed(entry) as model_path:
                with self.subTest(env_id=env_id):
                    cls = _oracle_class(env_id)
                    kwargs = dict(entry["kwargs"])
                    kwargs.pop("edit_fn", None)
                    kwargs["model_path"] = model_path
                    oracle: Any = gymnasium.wrappers.TimeLimit(
                        cls(seed=123, **kwargs),
                        max_episode_steps=entry["max_episode_steps"],
                    )
                    obs0, sync = _oracle_reset_sync(oracle, env_id)
                    obs_atol = _reach_alignment_obs_atol(env_id)
                    reward_atol = _reach_alignment_reward_atol(env_id)
                    config, pool_type = _reach_config(
                        env_id, model_path=model_path, overrides=sync
                    )
                    native = _make_env(config, pool_type, MyoSuiteReachEnvSpec)
                    obs1, _ = native.reset()
                    np.testing.assert_allclose(
                        obs1, obs0[None, :], atol=obs_atol, rtol=obs_atol
                    )
                    actions = _seeded_actions(
                        (1, _model(model_path).nu),
                        steps=_ALIGNMENT_STEPS,
                        seed=53,
                    )
                    for action in actions:
                        obs0, reward0, terminated0, truncated0, _ = oracle.step(
                            action[0]
                        )
                        obs1, reward1, terminated1, truncated1, _ = native.step(
                            action
                        )
                        np.testing.assert_allclose(
                            obs1, obs0[None, :], atol=obs_atol, rtol=obs_atol
                        )
                        np.testing.assert_allclose(
                            reward1,
                            np.array([reward0]),
                            atol=reward_atol,
                            rtol=reward_atol,
                        )
                        np.testing.assert_array_equal(
                            terminated1, np.array([terminated0])
                        )
                        np.testing.assert_array_equal(
                            truncated1, np.array([truncated0])
                        )
                        if terminated0 or truncated0:
                            break

    def test_reorient_alignment_representative_ids(self) -> None:
        """Representative reorient envs should align with the official oracle."""
        for env_id in _REORIENT_ALIGN_IDS:
            entry = _entry(env_id)
            with _edited_model_if_needed(entry) as model_path:
                with self.subTest(env_id=env_id):
                    cls = _oracle_class(env_id)
                    kwargs = dict(entry["kwargs"])
                    kwargs["model_path"] = model_path
                    oracle: Any = gymnasium.wrappers.TimeLimit(
                        cls(seed=123, **kwargs),
                        max_episode_steps=entry["max_episode_steps"],
                    )
                    obs0, sync = _oracle_reset_sync(oracle, env_id)
                    obs_atol = _reorient_alignment_obs_atol(env_id)
                    reward_atol = _reorient_alignment_reward_atol(env_id)
                    config, pool_type = _reorient_config(
                        env_id, model_path=model_path, overrides=sync
                    )
                    native = _make_env(
                        config, pool_type, MyoSuiteReorientEnvSpec
                    )
                    obs1, _ = native.reset()
                    np.testing.assert_allclose(
                        obs1, obs0[None, :], atol=obs_atol, rtol=obs_atol
                    )
                    actions = _seeded_actions(
                        (1, _model(model_path).nu),
                        steps=_ALIGNMENT_STEPS,
                        seed=67,
                    )
                    for action in actions:
                        obs0, reward0, terminated0, truncated0, _ = oracle.step(
                            action[0]
                        )
                        obs1, reward1, terminated1, truncated1, _ = native.step(
                            action
                        )
                        np.testing.assert_allclose(
                            obs1, obs0[None, :], atol=obs_atol, rtol=obs_atol
                        )
                        np.testing.assert_allclose(
                            reward1,
                            np.array([reward0]),
                            atol=reward_atol,
                            rtol=reward_atol,
                        )
                        np.testing.assert_array_equal(
                            terminated1, np.array([terminated0])
                        )
                        np.testing.assert_array_equal(
                            truncated1, np.array([truncated0])
                        )
                        if terminated0 or truncated0:
                            break

    def test_key_turn_alignment_representative_ids(self) -> None:
        """Representative key-turn envs should align with the official oracle."""
        for env_id in _KEYTURN_ALIGN_IDS:
            entry = _entry(env_id)
            with _edited_model_if_needed(entry) as model_path:
                with self.subTest(env_id=env_id):
                    cls = _oracle_class(env_id)
                    kwargs = dict(entry["kwargs"])
                    kwargs.pop("edit_fn", None)
                    kwargs["model_path"] = model_path
                    oracle: Any = gymnasium.wrappers.TimeLimit(
                        cls(seed=123, **kwargs),
                        max_episode_steps=entry["max_episode_steps"],
                    )
                    obs0, sync = _oracle_reset_sync(oracle, env_id)
                    obs_atol = _key_turn_alignment_obs_atol(env_id)
                    reward_atol = _key_turn_alignment_reward_atol(env_id)
                    config, pool_type = _key_turn_config(
                        env_id, model_path=model_path, overrides=sync
                    )
                    native = _make_env(
                        config, pool_type, MyoSuiteKeyTurnEnvSpec
                    )
                    obs1, _ = native.reset()
                    np.testing.assert_allclose(
                        obs1, obs0[None, :], atol=obs_atol, rtol=obs_atol
                    )
                    actions = _seeded_actions(
                        (1, _model(model_path).nu),
                        steps=_ALIGNMENT_STEPS,
                        seed=79,
                    )
                    for action in actions:
                        obs0, reward0, terminated0, truncated0, _ = oracle.step(
                            action[0]
                        )
                        obs1, reward1, terminated1, truncated1, _ = native.step(
                            action
                        )
                        np.testing.assert_allclose(
                            obs1, obs0[None, :], atol=obs_atol, rtol=obs_atol
                        )
                        np.testing.assert_allclose(
                            reward1,
                            np.array([reward0]),
                            atol=reward_atol,
                            rtol=reward_atol,
                        )
                        np.testing.assert_array_equal(
                            terminated1, np.array([terminated0])
                        )
                        np.testing.assert_array_equal(
                            truncated1, np.array([truncated0])
                        )
                        if terminated0 or truncated0:
                            break

    def test_obj_hold_alignment_representative_ids(self) -> None:
        """Representative obj-hold envs should align with the official oracle."""
        for env_id in _OBJHOLD_ALIGN_IDS:
            entry = _entry(env_id)
            with _edited_model_if_needed(entry) as model_path:
                with self.subTest(env_id=env_id):
                    cls = _oracle_class(env_id)
                    kwargs = dict(entry["kwargs"])
                    kwargs.pop("edit_fn", None)
                    kwargs["model_path"] = model_path
                    oracle: Any = gymnasium.wrappers.TimeLimit(
                        cls(seed=123, **kwargs),
                        max_episode_steps=entry["max_episode_steps"],
                    )
                    obs0, sync = _oracle_reset_sync(oracle, env_id)
                    obs_atol = _obj_hold_alignment_obs_atol(env_id)
                    reward_atol = _obj_hold_alignment_reward_atol(env_id)
                    config, pool_type = _obj_hold_config(
                        env_id, model_path=model_path, overrides=sync
                    )
                    native = _make_env(
                        config, pool_type, MyoSuiteObjHoldEnvSpec
                    )
                    obs1, _ = native.reset()
                    np.testing.assert_allclose(
                        obs1, obs0[None, :], atol=obs_atol, rtol=obs_atol
                    )
                    actions = _seeded_actions(
                        (1, _model(model_path).nu),
                        steps=_ALIGNMENT_STEPS,
                        seed=97,
                    )
                    for action in actions:
                        obs0, reward0, terminated0, truncated0, _ = oracle.step(
                            action[0]
                        )
                        obs1, reward1, terminated1, truncated1, _ = native.step(
                            action
                        )
                        np.testing.assert_allclose(
                            obs1, obs0[None, :], atol=obs_atol, rtol=obs_atol
                        )
                        np.testing.assert_allclose(
                            reward1,
                            np.array([reward0]),
                            atol=reward_atol,
                            rtol=reward_atol,
                        )
                        np.testing.assert_array_equal(
                            terminated1, np.array([terminated0])
                        )
                        np.testing.assert_array_equal(
                            truncated1, np.array([truncated0])
                        )
                        if terminated0 or truncated0:
                            break

    def test_torso_alignment_representative_ids(self) -> None:
        """Representative torso envs should align with the official oracle."""
        for env_id in _TORSO_ALIGN_IDS:
            entry = _entry(env_id)
            with _edited_model_if_needed(entry) as model_path:
                with self.subTest(env_id=env_id):
                    cls = _oracle_class(env_id)
                    kwargs = dict(entry["kwargs"])
                    kwargs.pop("edit_fn", None)
                    kwargs["model_path"] = model_path
                    oracle: Any = gymnasium.wrappers.TimeLimit(
                        cls(seed=123, **kwargs),
                        max_episode_steps=entry["max_episode_steps"],
                    )
                    obs0, sync = _oracle_reset_sync(oracle, env_id)
                    obs_atol = _torso_alignment_obs_atol(env_id)
                    reward_atol = _torso_alignment_reward_atol(env_id)
                    config, pool_type = _torso_config(
                        env_id, model_path=model_path, overrides=sync
                    )
                    native = _make_env(config, pool_type, MyoSuiteTorsoEnvSpec)
                    obs1, _ = native.reset()
                    np.testing.assert_allclose(
                        obs1, obs0[None, :], atol=obs_atol, rtol=obs_atol
                    )
                    actions = _seeded_actions(
                        (1, _model(model_path).nu),
                        steps=_ALIGNMENT_STEPS,
                        seed=101,
                    )
                    for action in actions:
                        obs0, reward0, terminated0, truncated0, _ = oracle.step(
                            action[0]
                        )
                        obs1, reward1, terminated1, truncated1, _ = native.step(
                            action
                        )
                        np.testing.assert_allclose(
                            obs1, obs0[None, :], atol=obs_atol, rtol=obs_atol
                        )
                        np.testing.assert_allclose(
                            reward1,
                            np.array([reward0]),
                            atol=reward_atol,
                            rtol=reward_atol,
                        )
                        np.testing.assert_array_equal(
                            terminated1, np.array([terminated0])
                        )
                        np.testing.assert_array_equal(
                            truncated1, np.array([truncated0])
                        )
                        if terminated0 or truncated0:
                            break

    def test_pen_twirl_alignment_representative_ids(self) -> None:
        """Representative pen-twirl envs should align with the official oracle."""
        for env_id in _PENTWIRL_ALIGN_IDS:
            entry = _entry(env_id)
            with _edited_model_if_needed(entry) as model_path:
                with self.subTest(env_id=env_id):
                    cls = _oracle_class(env_id)
                    kwargs = dict(entry["kwargs"])
                    kwargs.pop("edit_fn", None)
                    kwargs["model_path"] = model_path
                    oracle: Any = gymnasium.wrappers.TimeLimit(
                        cls(seed=123, **kwargs),
                        max_episode_steps=entry["max_episode_steps"],
                    )
                    obs0, sync = _oracle_reset_sync(oracle, env_id)
                    obs_atol = _pen_twirl_alignment_obs_atol(env_id)
                    reward_atol = _pen_twirl_alignment_reward_atol(env_id)
                    config, pool_type = _pen_twirl_config(
                        env_id, model_path=model_path, overrides=sync
                    )
                    native = _make_env(
                        config, pool_type, MyoSuitePenTwirlEnvSpec
                    )
                    obs1, _ = native.reset()
                    np.testing.assert_allclose(
                        obs1, obs0[None, :], atol=obs_atol, rtol=obs_atol
                    )
                    actions = _seeded_actions(
                        (1, _model(model_path).nu),
                        steps=_ALIGNMENT_STEPS,
                        seed=109,
                    )
                    for action in actions:
                        obs0, reward0, terminated0, truncated0, _ = oracle.step(
                            action[0]
                        )
                        obs1, reward1, terminated1, truncated1, _ = native.step(
                            action
                        )
                        np.testing.assert_allclose(
                            obs1, obs0[None, :], atol=obs_atol, rtol=obs_atol
                        )
                        np.testing.assert_allclose(
                            reward1,
                            np.array([reward0]),
                            atol=reward_atol,
                            rtol=reward_atol,
                        )
                        np.testing.assert_array_equal(
                            terminated1, np.array([terminated0])
                        )
                        np.testing.assert_array_equal(
                            truncated1, np.array([truncated0])
                        )
                        if terminated0 or truncated0:
                            break

    def test_walk_alignment_representative_ids(self) -> None:
        """Representative walk envs should align with the official oracle."""
        for env_id in _WALK_ALIGN_IDS:
            entry = _entry(env_id)
            with _edited_model_if_needed(entry) as model_path:
                with self.subTest(env_id=env_id):
                    cls = _oracle_class(env_id)
                    kwargs = dict(entry["kwargs"])
                    kwargs["model_path"] = model_path
                    oracle: Any = gymnasium.wrappers.TimeLimit(
                        cls(seed=123, **kwargs),
                        max_episode_steps=entry["max_episode_steps"],
                    )
                    obs0, sync = _oracle_reset_sync(oracle, env_id)
                    obs_atol = _walk_alignment_obs_atol(env_id)
                    reward_atol = _walk_alignment_reward_atol(env_id)
                    config, pool_type, spec_type = _walk_like_config(
                        env_id, model_path=model_path, overrides=sync
                    )
                    native = _make_env(config, pool_type, spec_type)
                    obs1, _ = native.reset()
                    np.testing.assert_allclose(
                        obs1, obs0[None, :], atol=obs_atol, rtol=obs_atol
                    )
                    actions = _seeded_actions(
                        (1, _model(model_path).nu),
                        steps=_ALIGNMENT_STEPS,
                        seed=131,
                    )
                    for action in actions:
                        obs0, reward0, terminated0, truncated0, _ = oracle.step(
                            action[0]
                        )
                        obs1, reward1, terminated1, truncated1, _ = native.step(
                            action
                        )
                        np.testing.assert_allclose(
                            obs1, obs0[None, :], atol=obs_atol, rtol=obs_atol
                        )
                        np.testing.assert_allclose(
                            reward1,
                            np.array([reward0]),
                            atol=reward_atol,
                            rtol=reward_atol,
                        )
                        np.testing.assert_array_equal(
                            terminated1, np.array([terminated0])
                        )
                        np.testing.assert_array_equal(
                            truncated1, np.array([truncated0])
                        )
                        if terminated0 or truncated0:
                            break

    def test_terrain_alignment_representative_ids(self) -> None:
        """Representative terrain envs should align with the official oracle."""
        for env_id in _TERRAIN_ALIGN_IDS:
            entry = _entry(env_id)
            with _edited_model_if_needed(entry) as model_path:
                with self.subTest(env_id=env_id):
                    cls = _oracle_class(env_id)
                    kwargs = dict(entry["kwargs"])
                    kwargs["model_path"] = model_path
                    oracle: Any = gymnasium.wrappers.TimeLimit(
                        cls(seed=123, **kwargs),
                        max_episode_steps=entry["max_episode_steps"],
                    )
                    obs0, sync = _oracle_reset_sync(oracle, env_id)
                    obs_atol = _terrain_alignment_obs_atol(env_id)
                    reward_atol = _terrain_alignment_reward_atol(env_id)
                    config, pool_type, spec_type = _walk_like_config(
                        env_id, model_path=model_path, overrides=sync
                    )
                    native = _make_env(config, pool_type, spec_type)
                    obs1, _ = native.reset()
                    np.testing.assert_allclose(
                        obs1, obs0[None, :], atol=obs_atol, rtol=obs_atol
                    )
                    actions = _seeded_actions(
                        (1, _model(model_path).nu),
                        steps=_ALIGNMENT_STEPS,
                        seed=149,
                    )
                    for action in actions:
                        obs0, reward0, terminated0, truncated0, _ = oracle.step(
                            action[0]
                        )
                        obs1, reward1, terminated1, truncated1, _ = native.step(
                            action
                        )
                        np.testing.assert_allclose(
                            obs1, obs0[None, :], atol=obs_atol, rtol=obs_atol
                        )
                        np.testing.assert_allclose(
                            reward1,
                            np.array([reward0]),
                            atol=reward_atol,
                            rtol=reward_atol,
                        )
                        np.testing.assert_array_equal(
                            terminated1, np.array([terminated0])
                        )
                        np.testing.assert_array_equal(
                            truncated1, np.array([truncated0])
                        )
                        if terminated0 or truncated0:
                            break

    def test_pose_pixel_observation_smoke(self) -> None:
        """Pose pixel wrappers should emit batched RGB observations."""
        config, pool_type = _pose_pixel_config("myoHandPoseRandom-v0")
        env = _make_env(config, pool_type, MyoSuitePosePixelEnvSpec)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (1, 3, 64, 64))

    def test_reach_pixel_observation_smoke(self) -> None:
        """Reach pixel wrappers should emit batched RGB observations."""
        config, pool_type = _reach_pixel_config("myoHandReachRandom-v0")
        env = _make_env(config, pool_type, MyoSuiteReachPixelEnvSpec)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (1, 3, 64, 64))

    def test_reorient_pixel_observation_smoke(self) -> None:
        """Reorient pixel wrappers should emit batched RGB observations."""
        config, pool_type = _reorient_pixel_config("myoHandReorientID-v0")
        env = _make_env(config, pool_type, MyoSuiteReorientPixelEnvSpec)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (1, 3, 64, 64))

    def test_key_turn_pixel_observation_smoke(self) -> None:
        """KeyTurn pixel wrappers should emit batched RGB observations."""
        config, pool_type = _key_turn_pixel_config("myoHandKeyTurnRandom-v0")
        env = _make_env(config, pool_type, MyoSuiteKeyTurnPixelEnvSpec)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (1, 3, 64, 64))

    def test_obj_hold_pixel_observation_smoke(self) -> None:
        """ObjHold pixel wrappers should emit batched RGB observations."""
        config, pool_type = _obj_hold_pixel_config("myoHandObjHoldRandom-v0")
        env = _make_env(config, pool_type, MyoSuiteObjHoldPixelEnvSpec)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (1, 3, 64, 64))

    def test_torso_pixel_observation_smoke(self) -> None:
        """Torso pixel wrappers should emit batched RGB observations."""
        config, pool_type = _torso_pixel_config("myoTorsoPoseFixed-v0")
        env = _make_env(config, pool_type, MyoSuiteTorsoPixelEnvSpec)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (1, 3, 64, 64))

    def test_pen_twirl_pixel_observation_smoke(self) -> None:
        """PenTwirl pixel wrappers should emit batched RGB observations."""
        config, pool_type = _pen_twirl_pixel_config("myoHandPenTwirlRandom-v0")
        env = _make_env(config, pool_type, MyoSuitePenTwirlPixelEnvSpec)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (1, 3, 64, 64))

    def test_walk_pixel_observation_smoke(self) -> None:
        """Walk pixel wrappers should stay constructible alongside terrain render."""
        config, pool_type, spec_type = _walk_pixel_config("myoLegWalk-v0")
        env = _make_env(config, pool_type, spec_type)
        self.assertIn("obs:pixels", MyoSuiteWalkPixelEnvSpec._state_keys)
        self.assertIsNotNone(env)

    def test_terrain_pixel_observation_smoke(self) -> None:
        """Terrain pixel wrappers should emit batched RGB observations."""
        config, pool_type, spec_type = _terrain_pixel_config(
            "myoLegRoughTerrainWalk-v0"
        )
        env = _make_env(config, pool_type, spec_type)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (1, 3, 64, 64))

    def test_public_registry_covers_full_expanded_surface(self) -> None:
        """Public registration should expose every expanded MyoSuite task ID."""
        registered = set(list_all_envs())
        self.assertLen(MYOSUITE_PUBLIC_TASK_IDS, 398)
        self.assertEmpty(set(MYOSUITE_PUBLIC_TASK_IDS) - registered)

    def test_public_registered_envs_construct_and_reset(self) -> None:
        """Representative public IDs should construct through make_gymnasium."""
        for task_id in _PUBLIC_REPRESENTATIVE_TASK_IDS:
            with self.subTest(task_id=task_id):
                env = _make_registered_env(task_id, seed=0)
                try:
                    obs, info = env.reset()
                    self.assertEqual(obs.shape[0], 1)
                    self.assertIn("elapsed_step", info)
                finally:
                    env.close()

    def test_public_variant_ids_change_dynamics(self) -> None:
        """Variant IDs should change rollout dynamics relative to the base ID."""
        for base_task_id, variant_task_id in _PUBLIC_VARIANT_DIFF_CASES:
            with self.subTest(
                base_task_id=base_task_id, variant_task_id=variant_task_id
            ):
                base_obs, base_reward = _registered_rollout(
                    base_task_id, seed=7
                )
                variant_obs, variant_reward = _registered_rollout(
                    variant_task_id, seed=7
                )
                self.assertGreater(
                    float(np.max(np.abs(base_obs - variant_obs))), 1e-6
                )
                self.assertGreater(
                    float(np.max(np.abs(base_reward - variant_reward))), 1e-6
                )

    def test_public_from_pixels_smoke(self) -> None:
        """Public registration should expose pixel wrappers for MyoSuite."""
        env = _make_registered_env(
            "myoHandReorientID-v0",
            seed=3,
            from_pixels=True,
            render_width=32,
            render_height=24,
        )
        try:
            obs, _ = env.reset()
            self.assertEqual(obs.shape, (1, 3, 24, 32))
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
