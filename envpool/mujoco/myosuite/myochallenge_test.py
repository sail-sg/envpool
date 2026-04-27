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
"""Internal MyoSuite MyoChallenge native env tests."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import gymnasium
import mujoco
import numpy as np
from absl.testing import absltest

from envpool.mujoco.myosuite.config import (
    myosuite_expanded_entry,
    resolve_myosuite_model_path,
)
from envpool.mujoco.myosuite.metadata import MYOSUITE_DIRECT_ENTRIES
from envpool.mujoco.myosuite.native import (
    MyoChallengeBaodingEnvSpec,
    MyoChallengeBaodingGymnasiumEnvPool,
    MyoChallengeBaodingPixelEnvSpec,
    MyoChallengeBaodingPixelGymnasiumEnvPool,
    MyoChallengeBimanualEnvSpec,
    MyoChallengeBimanualGymnasiumEnvPool,
    MyoChallengeBimanualPixelEnvSpec,
    MyoChallengeBimanualPixelGymnasiumEnvPool,
    MyoChallengeChaseTagEnvSpec,
    MyoChallengeChaseTagGymnasiumEnvPool,
    MyoChallengeChaseTagPixelEnvSpec,
    MyoChallengeChaseTagPixelGymnasiumEnvPool,
    MyoChallengeRelocateEnvSpec,
    MyoChallengeRelocateGymnasiumEnvPool,
    MyoChallengeRelocatePixelEnvSpec,
    MyoChallengeRelocatePixelGymnasiumEnvPool,
    MyoChallengeReorientEnvSpec,
    MyoChallengeReorientGymnasiumEnvPool,
    MyoChallengeReorientPixelEnvSpec,
    MyoChallengeReorientPixelGymnasiumEnvPool,
    MyoChallengeRunTrackEnvSpec,
    MyoChallengeRunTrackGymnasiumEnvPool,
    MyoChallengeRunTrackPixelEnvSpec,
    MyoChallengeRunTrackPixelGymnasiumEnvPool,
    MyoChallengeSoccerEnvSpec,
    MyoChallengeSoccerGymnasiumEnvPool,
    MyoChallengeSoccerPixelEnvSpec,
    MyoChallengeSoccerPixelGymnasiumEnvPool,
    MyoChallengeTableTennisEnvSpec,
    MyoChallengeTableTennisGymnasiumEnvPool,
    MyoChallengeTableTennisPixelEnvSpec,
    MyoChallengeTableTennisPixelGymnasiumEnvPool,
)
from envpool.mujoco.myosuite.oracle_utils import load_oracle_class
from envpool.mujoco.myosuite.paths import (
    myosuite_asset_root,
)
from envpool.python.glfw_context import preload_windows_gl_dlls

preload_windows_gl_dlls(strict=True)

_REORIENT_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["suite"] == "myochallenge"
    and entry["class_name"] == "ReorientEnvV0"
)
_RELOCATE_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["suite"] == "myochallenge"
    and entry["class_name"] == "RelocateEnvV0"
)
_BAODING_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["suite"] == "myochallenge"
    and entry["class_name"] == "BaodingEnvV1"
)
_BIMANUAL_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["suite"] == "myochallenge"
    and entry["class_name"] == "BimanualEnvV1"
)
_RUNTRACK_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["suite"] == "myochallenge" and entry["class_name"] == "RunTrack"
)
_SOCCER_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["suite"] == "myochallenge" and entry["class_name"] == "SoccerEnvV0"
)
_CHASETAG_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["suite"] == "myochallenge"
    and entry["class_name"] == "ChaseTagEnvV0"
)
_TABLETENNIS_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["suite"] == "myochallenge"
    and entry["class_name"] == "TableTennisEnvV0"
)
_ALIGNMENT_STEPS = 32
_REORIENT_ALIGNMENT_OBS_ATOL = 2e-3
_REORIENT_ALIGNMENT_ROT_ATOL = 2e-2
_DEFAULT_ALIGNMENT_ATOL = 1e-5
_LOCOMOTION_ALIGNMENT_ATOL = 1e-4
_BIMANUAL_ALIGNMENT_ATOL = 5e-4
_TABLETENNIS_ALIGNMENT_ATOL = 5e-4


def _entry(env_id: str) -> dict[str, Any]:
    entry, variant_kwargs = myosuite_expanded_entry(env_id)
    merged = dict(entry)
    merged["id"] = env_id
    merged["kwargs"] = {**entry["kwargs"], **variant_kwargs}
    return merged


def _runtime_model_path(model_path: str) -> Path:
    path = Path(resolve_myosuite_model_path(model_path))
    return path if path.is_absolute() else myosuite_asset_root() / path


def _oracle_model_path(model_path: str) -> Path:
    path = Path(model_path)
    return path if path.is_absolute() else myosuite_asset_root() / path


@cache
def _model(path: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(_runtime_model_path(path)))


def _make_env(config: tuple[Any, ...], pool_type: type, spec_type: type) -> Any:
    return pool_type(spec_type(config))


def _soccer_policy_id(policy: str) -> int:
    return {"block_ball": 0, "random": 1, "stationary": 2}[policy]


def _chasetag_policy_id(policy: str) -> int:
    return {
        "static_stationary": 0,
        "stationary": 1,
        "random": 2,
        "chase_player": 3,
        "repeller": 4,
    }[policy]


def _runtrack_osl_state_id(state: str) -> int:
    return {
        "e_stance": 0,
        "l_stance": 1,
        "e_swing": 2,
        "l_swing": 3,
    }[state]


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


def _batched_action_shape(env: Any) -> tuple[int, int]:
    return (1, int(env.action_space.shape[-1]))


def _relocate_alignment_obs_atol(env_id: str) -> float:
    if env_id == "myoChallengeRelocateP1-v0":
        return 2e-2
    return 2e-3


def _reorient_alignment_reward_atol() -> float:
    # Reorient dense reward scales `pos_dist` by 100x, so the accepted
    # observation residual on pos_err alone can legitimately amplify into a
    # ~2e-1 scalar reward drift.
    return 2e-1


def _assert_reorient_obs_close(
    actual: np.ndarray, expected: np.ndarray
) -> None:
    # The only residual after reset-sync is the `mat2euler(site_xmat)` block:
    # obj_rot, goal_rot, and rot_err. Keep the rest of the observation tight.
    np.testing.assert_allclose(
        actual[:54],
        expected[:54],
        atol=_REORIENT_ALIGNMENT_OBS_ATOL,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        actual[54:63],
        expected[54:63],
        atol=_REORIENT_ALIGNMENT_ROT_ATOL,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        actual[63:],
        expected[63:],
        atol=_REORIENT_ALIGNMENT_OBS_ATOL,
        rtol=1e-5,
    )


def _relocate_alignment_reward_atol(env_id: str) -> float:
    if env_id == "myoChallengeRelocateP1-v0":
        return 1.0
    return 2e-2


def _reorient_config(
    env_id: str, overrides: dict[str, Any] | None = None
) -> tuple[tuple[Any, ...], type, type]:
    entry = _entry(env_id)
    kwargs = dict(entry["kwargs"])
    model = _model(kwargs["model_path"])
    obj_mass = kwargs.get("obj_mass_range", [0.108, 0.108])
    config = MyoChallengeReorientEnvSpec.gen_config(
        num_envs=1,
        batch_size=1,
        max_num_players=1,
        frame_skip=int(kwargs.get("frame_skip", 5)),
        model_path=str(kwargs["model_path"]),
        normalize_act=bool(kwargs.get("normalize_act", True)),
        muscle_condition=str(kwargs.get("muscle_condition", "")),
        obs_dim=(model.nq - 7) + (model.nv - 6) + 18 + model.na,
        qpos_dim=model.nq,
        qvel_dim=model.nv,
        act_dim=model.na,
        action_dim=model.nu,
        goal_pos_low=float(kwargs.get("goal_pos", (0.0, 0.0))[0]),
        goal_pos_high=float(kwargs.get("goal_pos", (0.0, 0.0))[1]),
        goal_rot_low=float(kwargs.get("goal_rot", (0.0, 0.0))[0]),
        goal_rot_high=float(kwargs.get("goal_rot", (0.0, 0.0))[1]),
        obj_size_change=float(kwargs.get("obj_size_change", 0.0)),
        obj_mass_low=float(obj_mass[0]),
        obj_mass_high=float(obj_mass[1]),
        obj_friction_change=list(
            kwargs.get("obj_friction_change", (0.0, 0.0, 0.0))
        ),
        pos_th=float(kwargs.get("pos_th", 0.025)),
        rot_th=float(kwargs.get("rot_th", 0.262)),
        drop_th=float(kwargs.get("drop_th", 0.2)),
        reward_pos_dist_w=float(
            kwargs.get("weighted_reward_keys", {}).get("pos_dist", 100.0)
        ),
        reward_rot_dist_w=float(
            kwargs.get("weighted_reward_keys", {}).get("rot_dist", 1.0)
        ),
        reward_bonus_w=float(
            kwargs.get("weighted_reward_keys", {}).get("bonus", 0.0)
        ),
        reward_act_reg_w=float(
            kwargs.get("weighted_reward_keys", {}).get("act_reg", 0.0)
        ),
        reward_penalty_w=float(
            kwargs.get("weighted_reward_keys", {}).get("penalty", 0.0)
        ),
    )
    if overrides:
        config = MyoChallengeReorientEnvSpec.gen_config(
            **dict(
                zip(
                    MyoChallengeReorientEnvSpec._config_keys,
                    config,
                    strict=False,
                ),
                **overrides,
            )
        )
    return (
        config,
        MyoChallengeReorientGymnasiumEnvPool,
        MyoChallengeReorientEnvSpec,
    )


def _relocate_config(
    env_id: str, overrides: dict[str, Any] | None = None
) -> tuple[tuple[Any, ...], type, type]:
    entry = _entry(env_id)
    kwargs = dict(entry["kwargs"])
    model = _model(kwargs["model_path"])
    target_xyz_range = kwargs["target_xyz_range"]
    target_rot_range = kwargs["target_rxryrz_range"]
    obj_xyz_range = kwargs.get("obj_xyz_range")
    obj_geom_range = kwargs.get("obj_geom_range")
    obj_mass_range = kwargs.get("obj_mass_range")
    obj_friction_range = kwargs.get("obj_friction_range")
    config = MyoChallengeRelocateEnvSpec.gen_config(
        num_envs=1,
        batch_size=1,
        max_num_players=1,
        frame_skip=int(kwargs.get("frame_skip", 5)),
        model_path=str(kwargs["model_path"]),
        normalize_act=bool(kwargs.get("normalize_act", True)),
        muscle_condition=str(kwargs.get("muscle_condition", "")),
        obs_dim=(model.nq - 7) + (model.nv - 6) + 18 + model.na,
        qpos_dim=model.nq,
        qvel_dim=model.nv,
        act_dim=model.na,
        action_dim=model.nu,
        target_xyz_low=list(target_xyz_range["low"]),
        target_xyz_high=list(target_xyz_range["high"]),
        target_rxryrz_low=list(target_rot_range["low"]),
        target_rxryrz_high=list(target_rot_range["high"]),
        obj_xyz_low=[] if obj_xyz_range is None else list(obj_xyz_range["low"]),
        obj_xyz_high=[]
        if obj_xyz_range is None
        else list(obj_xyz_range["high"]),
        obj_geom_low=[]
        if obj_geom_range is None
        else list(obj_geom_range["low"]),
        obj_geom_high=[]
        if obj_geom_range is None
        else list(obj_geom_range["high"]),
        obj_mass_low=0.0
        if obj_mass_range is None
        else float(obj_mass_range["low"]),
        obj_mass_high=0.0
        if obj_mass_range is None
        else float(obj_mass_range["high"]),
        obj_friction_low=[]
        if obj_friction_range is None
        else list(obj_friction_range["low"]),
        obj_friction_high=[]
        if obj_friction_range is None
        else list(obj_friction_range["high"]),
        qpos_noise_range=float(kwargs.get("qpos_noise_range", 0.0)),
        pos_th=float(kwargs.get("pos_th", 0.025)),
        rot_th=float(kwargs.get("rot_th", 0.262)),
        drop_th=float(kwargs.get("drop_th", 0.5)),
        reward_pos_dist_w=float(
            kwargs.get("weighted_reward_keys", {}).get("pos_dist", 100.0)
        ),
        reward_rot_dist_w=float(
            kwargs.get("weighted_reward_keys", {}).get("rot_dist", 1.0)
        ),
        reward_act_reg_w=float(
            kwargs.get("weighted_reward_keys", {}).get("act_reg", 0.0)
        ),
    )
    if overrides:
        config = MyoChallengeRelocateEnvSpec.gen_config(
            **dict(
                zip(
                    MyoChallengeRelocateEnvSpec._config_keys,
                    config,
                    strict=False,
                ),
                **overrides,
            )
        )
    return (
        config,
        MyoChallengeRelocateGymnasiumEnvPool,
        MyoChallengeRelocateEnvSpec,
    )


def _reorient_pixel_config(env_id: str) -> tuple[tuple[Any, ...], type, type]:
    config, _, _ = _reorient_config(env_id)
    values = dict(
        zip(MyoChallengeReorientEnvSpec._config_keys, config, strict=False)
    )
    pixel = MyoChallengeReorientPixelEnvSpec.gen_config(
        **values,
        render_width=64,
        render_height=64,
        render_camera_id=-1,
    )
    return (
        pixel,
        MyoChallengeReorientPixelGymnasiumEnvPool,
        MyoChallengeReorientPixelEnvSpec,
    )


def _relocate_pixel_config(env_id: str) -> tuple[tuple[Any, ...], type, type]:
    config, _, _ = _relocate_config(env_id)
    values = dict(
        zip(MyoChallengeRelocateEnvSpec._config_keys, config, strict=False)
    )
    pixel = MyoChallengeRelocatePixelEnvSpec.gen_config(
        **values,
        render_width=64,
        render_height=64,
        render_camera_id=-1,
    )
    return (
        pixel,
        MyoChallengeRelocatePixelGymnasiumEnvPool,
        MyoChallengeRelocatePixelEnvSpec,
    )


def _baoding_config(
    env_id: str, overrides: dict[str, Any] | None = None
) -> tuple[tuple[Any, ...], type, type]:
    entry = _entry(env_id)
    kwargs = dict(entry["kwargs"])
    model = _model(kwargs["model_path"])
    ball1_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball1")
    obj_size_range = kwargs.get("obj_size_range")
    obj_mass_range = kwargs.get("obj_mass_range")
    obj_friction_change = kwargs.get("obj_friction_change")
    config = MyoChallengeBaodingEnvSpec.gen_config(
        num_envs=1,
        batch_size=1,
        max_num_players=1,
        frame_skip=int(kwargs.get("frame_skip", 10)),
        model_path=str(kwargs["model_path"]),
        normalize_act=bool(kwargs.get("normalize_act", True)),
        muscle_condition=str(kwargs.get("muscle_condition", "")),
        obs_dim=(model.nq - 14) + 24 + model.na,
        qpos_dim=model.nq,
        qvel_dim=model.nv,
        act_dim=model.na,
        action_dim=model.nu,
        drop_th=float(kwargs.get("drop_th", 1.25)),
        proximity_th=float(kwargs.get("proximity_th", 0.015)),
        goal_time_period_low=float(kwargs.get("goal_time_period", (5, 5))[0]),
        goal_time_period_high=float(kwargs.get("goal_time_period", (5, 5))[1]),
        goal_xrange_low=float(kwargs.get("goal_xrange", (0.025, 0.025))[0]),
        goal_xrange_high=float(kwargs.get("goal_xrange", (0.025, 0.025))[1]),
        goal_yrange_low=float(kwargs.get("goal_yrange", (0.028, 0.028))[0]),
        goal_yrange_high=float(kwargs.get("goal_yrange", (0.028, 0.028))[1]),
        task_choice=str(kwargs.get("task_choice", "fixed")),
        fixed_task=2,
        reward_pos_dist_1_w=float(
            kwargs.get("weighted_reward_keys", {}).get("pos_dist_1", 5.0)
        ),
        reward_pos_dist_2_w=float(
            kwargs.get("weighted_reward_keys", {}).get("pos_dist_2", 5.0)
        ),
        obj_size_low=0.0
        if obj_size_range is None
        else float(obj_size_range[0]),
        obj_size_high=0.0
        if obj_size_range is None
        else float(obj_size_range[1]),
        obj_mass_low=0.0
        if obj_mass_range is None
        else float(obj_mass_range[0]),
        obj_mass_high=0.0
        if obj_mass_range is None
        else float(obj_mass_range[1]),
        obj_friction_low=[]
        if obj_friction_change is None
        else list(
            (
                model.geom_friction[ball1_gid] - np.asarray(obj_friction_change)
            ).tolist()
        ),
        obj_friction_high=[]
        if obj_friction_change is None
        else list(
            (
                model.geom_friction[ball1_gid] + np.asarray(obj_friction_change)
            ).tolist()
        ),
    )
    if overrides:
        config = MyoChallengeBaodingEnvSpec.gen_config(
            **dict(
                zip(
                    MyoChallengeBaodingEnvSpec._config_keys,
                    config,
                    strict=False,
                ),
                **overrides,
            )
        )
    return (
        config,
        MyoChallengeBaodingGymnasiumEnvPool,
        MyoChallengeBaodingEnvSpec,
    )


def _baoding_pixel_config(env_id: str) -> tuple[tuple[Any, ...], type, type]:
    config, _, _ = _baoding_config(env_id)
    values = dict(
        zip(MyoChallengeBaodingEnvSpec._config_keys, config, strict=False)
    )
    pixel = MyoChallengeBaodingPixelEnvSpec.gen_config(
        **values,
        render_width=64,
        render_height=64,
        render_camera_id=-1,
    )
    return (
        pixel,
        MyoChallengeBaodingPixelGymnasiumEnvPool,
        MyoChallengeBaodingPixelEnvSpec,
    )


def _bimanual_index_sets(model: mujoco.MjModel) -> tuple[list[int], ...]:
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


def _bimanual_config(
    env_id: str, overrides: dict[str, Any] | None = None
) -> tuple[tuple[Any, ...], type, type]:
    entry = _entry(env_id)
    kwargs = dict(entry["kwargs"])
    model = _model(kwargs["model_path"])
    myo_qpos, myo_dof, prosth_qpos, prosth_dof, _, _ = _bimanual_index_sets(
        model
    )
    obj_bid = model.body("manip_object").id
    obj_gid = model.body(obj_bid).geomadr + 1
    base_friction = np.asarray(model.geom_friction[obj_gid]).reshape(-1)
    obj_mass_change = kwargs.get("obj_mass_change")
    obj_friction_change = kwargs.get("obj_friction_change")
    config = MyoChallengeBimanualEnvSpec.gen_config(
        num_envs=1,
        batch_size=1,
        max_num_players=1,
        frame_skip=int(kwargs.get("frame_skip", 5)),
        model_path=str(kwargs["model_path"]),
        normalize_act=bool(kwargs.get("normalize_act", True)),
        muscle_condition=str(kwargs.get("muscle_condition", "")),
        obs_dim=1
        + len(myo_qpos)
        + len(myo_dof)
        + len(prosth_qpos)
        + len(prosth_dof)
        + 7
        + 6
        + 5
        + model.na,
        qpos_dim=model.nq,
        qvel_dim=model.nv,
        act_dim=model.na,
        action_dim=model.nu,
        proximity_th=0.17,
        start_center=[-0.4, -0.25, 1.05],
        goal_center=[0.4, -0.25, 1.05],
        start_shifts=[0.055, 0.055, 0.0],
        goal_shifts=[0.098, 0.098, 0.0],
        reward_reach_dist_w=float(
            kwargs.get("weighted_reward_keys", {}).get("reach_dist", -0.1)
        ),
        reward_act_w=float(
            kwargs.get("weighted_reward_keys", {}).get("act", 0.0)
        ),
        reward_fin_dis_w=float(
            kwargs.get("weighted_reward_keys", {}).get("fin_dis", -0.5)
        ),
        reward_pass_err_w=float(
            kwargs.get("weighted_reward_keys", {}).get("pass_err", -1.0)
        ),
        obj_scale_change=list(kwargs.get("obj_scale_change", [])),
        obj_mass_low=0.0
        if obj_mass_change is None
        else float(model.body_mass[obj_bid] + obj_mass_change[0]),
        obj_mass_high=0.0
        if obj_mass_change is None
        else float(model.body_mass[obj_bid] + obj_mass_change[1]),
        obj_friction_low=[]
        if obj_friction_change is None
        else list((base_friction - np.asarray(obj_friction_change)).tolist()),
        obj_friction_high=[]
        if obj_friction_change is None
        else list((base_friction + np.asarray(obj_friction_change)).tolist()),
    )
    if overrides:
        config = MyoChallengeBimanualEnvSpec.gen_config(
            **dict(
                zip(
                    MyoChallengeBimanualEnvSpec._config_keys,
                    config,
                    strict=False,
                ),
                **overrides,
            )
        )
    return (
        config,
        MyoChallengeBimanualGymnasiumEnvPool,
        MyoChallengeBimanualEnvSpec,
    )


def _bimanual_pixel_config(env_id: str) -> tuple[tuple[Any, ...], type, type]:
    config, _, _ = _bimanual_config(env_id)
    values = dict(
        zip(MyoChallengeBimanualEnvSpec._config_keys, config, strict=False)
    )
    pixel = MyoChallengeBimanualPixelEnvSpec.gen_config(
        **values,
        render_width=64,
        render_height=64,
        render_camera_id=-1,
    )
    return (
        pixel,
        MyoChallengeBimanualPixelGymnasiumEnvPool,
        MyoChallengeBimanualPixelEnvSpec,
    )


def _runtrack_config(
    env_id: str, overrides: dict[str, Any] | None = None
) -> tuple[tuple[Any, ...], type, type]:
    entry = _entry(env_id)
    kwargs = dict(entry["kwargs"])
    model = _model(kwargs["model_path"])
    obs_dim = 17 + 17 + 2 + 4 + model.na * 4 + 2 + 2
    config = MyoChallengeRunTrackEnvSpec.gen_config(
        num_envs=1,
        batch_size=1,
        max_num_players=1,
        frame_skip=int(kwargs.get("frame_skip", 5)),
        model_path=str(kwargs["model_path"]),
        normalize_act=bool(kwargs.get("normalize_act", True)),
        muscle_condition=str(kwargs.get("muscle_condition", "")),
        obs_dim=obs_dim,
        qpos_dim=model.nq,
        qvel_dim=model.nv,
        act_dim=model.na,
        action_dim=model.na,
        ctrl_dim=model.nu,
        reset_type=str(kwargs.get("reset_type", "random")),
        terrain=str(kwargs.get("terrain", "flat")),
        start_pos=float(kwargs.get("start_pos", 14)),
        end_pos=float(kwargs.get("end_pos", -15)),
        real_width=float(kwargs.get("real_width", 1)),
        hills_difficulties=list(kwargs.get("hills_difficulties", [])),
        rough_difficulties=list(kwargs.get("rough_difficulties", [])),
        stairs_difficulties=list(kwargs.get("stairs_difficulties", [])),
        reward_sparse_w=1.0,
        reward_solved_w=10.0,
    )
    if overrides:
        config = MyoChallengeRunTrackEnvSpec.gen_config(
            **dict(
                zip(
                    MyoChallengeRunTrackEnvSpec._config_keys,
                    config,
                    strict=False,
                ),
                **overrides,
            )
        )
    return (
        config,
        MyoChallengeRunTrackGymnasiumEnvPool,
        MyoChallengeRunTrackEnvSpec,
    )


def _runtrack_pixel_config(env_id: str) -> tuple[tuple[Any, ...], type, type]:
    config, _, _ = _runtrack_config(env_id)
    values = dict(
        zip(MyoChallengeRunTrackEnvSpec._config_keys, config, strict=False)
    )
    pixel = MyoChallengeRunTrackPixelEnvSpec.gen_config(
        **values,
        render_width=64,
        render_height=64,
        render_camera_id=-1,
    )
    return (
        pixel,
        MyoChallengeRunTrackPixelGymnasiumEnvPool,
        MyoChallengeRunTrackPixelEnvSpec,
    )


def _soccer_config(
    env_id: str, overrides: dict[str, Any] | None = None
) -> tuple[tuple[Any, ...], type, type]:
    entry = _entry(env_id)
    kwargs = dict(entry["kwargs"])
    model = _model(kwargs["model_path"])
    internal_joint_count = sum(
        1
        for joint_id in range(model.njnt)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        not in (None, "root")
    )
    obs_dim = (
        internal_joint_count
        + internal_joint_count
        + 4
        + 4
        + 3
        + 7
        + 6
        + model.na * 4
    )
    probabilities = kwargs.get("goalkeeper_probabilities", [0.1, 0.45, 0.45])
    config = MyoChallengeSoccerEnvSpec.gen_config(
        num_envs=1,
        batch_size=1,
        max_num_players=1,
        frame_skip=int(kwargs.get("frame_skip", 10)),
        model_path=str(kwargs["model_path"]),
        normalize_act=bool(kwargs.get("normalize_act", True)),
        muscle_condition=str(kwargs.get("muscle_condition", "")),
        obs_dim=obs_dim,
        qpos_dim=model.nq,
        qvel_dim=model.nv,
        act_dim=model.na,
        action_dim=model.nu,
        reset_type=str(kwargs.get("reset_type", "none")),
        min_agent_spawn_distance=float(
            kwargs.get("min_agent_spawn_distance", 1)
        ),
        random_vel_low=float(kwargs.get("random_vel_range", [1.0, 5.0])[0]),
        random_vel_high=float(kwargs.get("random_vel_range", [1.0, 5.0])[1]),
        rnd_pos_noise=float(kwargs.get("rnd_pos_noise", 1.0)),
        rnd_joint_noise=float(kwargs.get("rnd_joint_noise", 0.02)),
        goalkeeper_probabilities=list(probabilities),
        max_time_sec=float(kwargs.get("max_time_sec", 10)),
        reward_goal_scored_w=1000.0,
        reward_time_cost_w=-0.01,
        reward_act_reg_w=-100.0,
        reward_pain_w=-10.0,
    )
    if overrides:
        config = MyoChallengeSoccerEnvSpec.gen_config(
            **dict(
                zip(
                    MyoChallengeSoccerEnvSpec._config_keys, config, strict=False
                ),
                **overrides,
            )
        )
    return (
        config,
        MyoChallengeSoccerGymnasiumEnvPool,
        MyoChallengeSoccerEnvSpec,
    )


def _soccer_pixel_config(env_id: str) -> tuple[tuple[Any, ...], type, type]:
    config, _, _ = _soccer_config(env_id)
    values = dict(
        zip(MyoChallengeSoccerEnvSpec._config_keys, config, strict=False)
    )
    pixel = MyoChallengeSoccerPixelEnvSpec.gen_config(
        **values,
        render_width=64,
        render_height=64,
        render_camera_id=-1,
    )
    return (
        pixel,
        MyoChallengeSoccerPixelGymnasiumEnvPool,
        MyoChallengeSoccerPixelEnvSpec,
    )


def _chasetag_config(
    env_id: str, overrides: dict[str, Any] | None = None
) -> tuple[tuple[Any, ...], type, type]:
    entry = _entry(env_id)
    kwargs = dict(entry["kwargs"])
    model = _model(kwargs["model_path"])
    obs_dim = 28 + 28 + 4 + 4 + 3 + 2 + 2 + 2 + model.na * 4
    config = MyoChallengeChaseTagEnvSpec.gen_config(
        num_envs=1,
        batch_size=1,
        max_num_players=1,
        frame_skip=int(kwargs.get("frame_skip", 10)),
        model_path=str(kwargs["model_path"]),
        normalize_act=bool(kwargs.get("normalize_act", True)),
        muscle_condition=str(kwargs.get("muscle_condition", "")),
        obs_dim=obs_dim,
        qpos_dim=model.nq,
        qvel_dim=model.nv,
        act_dim=model.na,
        action_dim=model.nu,
        reset_type=str(kwargs.get("reset_type", "init")),
        win_distance=float(kwargs.get("win_distance", 0.5)),
        min_spawn_distance=float(kwargs.get("min_spawn_distance", 2)),
        task_choice=str(kwargs.get("task_choice", "CHASE")),
        terrain=str(kwargs.get("terrain", "FLAT")),
        repeller_opponent=bool(kwargs.get("repeller_opponent", False)),
        hills_range=list(kwargs.get("hills_range", [0.0, 0.0])),
        rough_range=list(kwargs.get("rough_range", [0.0, 0.0])),
        relief_range=list(kwargs.get("relief_range", [0.0, 0.0])),
        chase_vel_low=float(kwargs.get("chase_vel_range", [1.0, 1.0])[0]),
        chase_vel_high=float(kwargs.get("chase_vel_range", [1.0, 1.0])[1]),
        random_vel_low=float(kwargs.get("random_vel_range", [1.0, 1.0])[0]),
        random_vel_high=float(kwargs.get("random_vel_range", [1.0, 1.0])[1]),
        repeller_vel_low=float(kwargs.get("repeller_vel_range", [1.0, 1.0])[0]),
        repeller_vel_high=float(
            kwargs.get("repeller_vel_range", [1.0, 1.0])[1]
        ),
        opponent_probabilities=list(
            kwargs.get("opponent_probabilities", [0.1, 0.45, 0.45])
        ),
        reward_distance_w=-0.1,
        reward_lose_w=-1000.0,
    )
    if overrides:
        config = MyoChallengeChaseTagEnvSpec.gen_config(
            **dict(
                zip(
                    MyoChallengeChaseTagEnvSpec._config_keys,
                    config,
                    strict=False,
                ),
                **overrides,
            )
        )
    return (
        config,
        MyoChallengeChaseTagGymnasiumEnvPool,
        MyoChallengeChaseTagEnvSpec,
    )


def _chasetag_pixel_config(env_id: str) -> tuple[tuple[Any, ...], type, type]:
    config, _, _ = _chasetag_config(env_id)
    values = dict(
        zip(MyoChallengeChaseTagEnvSpec._config_keys, config, strict=False)
    )
    pixel = MyoChallengeChaseTagPixelEnvSpec.gen_config(
        **values,
        render_width=64,
        render_height=64,
        render_camera_id=-1,
    )
    return (
        pixel,
        MyoChallengeChaseTagPixelGymnasiumEnvPool,
        MyoChallengeChaseTagPixelEnvSpec,
    )


def _tabletennis_config(
    env_id: str, overrides: dict[str, Any] | None = None
) -> tuple[tuple[Any, ...], type, type]:
    entry = _entry(env_id)
    kwargs = dict(entry["kwargs"])
    model_path = str(resolve_myosuite_model_path(kwargs["model_path"]))
    model = _model(kwargs["model_path"])
    body_qpos_count = sum(
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
            and name != "pingpong_freejoint"
            and name != "paddle_freejoint"
        )
    )
    body_dof_count = sum(
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
    obs_dim = (
        3
        + body_qpos_count
        + body_dof_count
        + 3
        + 3
        + 3
        + 3
        + 4
        + 3
        + 6
        + model.na
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
    config = MyoChallengeTableTennisEnvSpec.gen_config(
        num_envs=1,
        batch_size=1,
        max_num_players=1,
        frame_skip=int(kwargs.get("frame_skip", 5)),
        model_path=model_path,
        normalize_act=bool(kwargs.get("normalize_act", True)),
        muscle_condition=str(kwargs.get("muscle_condition", "")),
        obs_dim=obs_dim,
        qpos_dim=model.nq,
        qvel_dim=model.nv,
        act_dim=model.na,
        action_dim=model.nu,
        ball_xyz_low=list(ball_xyz_range.get("low", [])),
        ball_xyz_high=list(ball_xyz_range.get("high", [])),
        ball_qvel=bool(kwargs.get("ball_qvel", False)),
        ball_friction_low=list(ball_friction_range.get("low", [])),
        ball_friction_high=list(ball_friction_range.get("high", [])),
        paddle_mass_low=float(paddle_mass_range[0]),
        paddle_mass_high=float(paddle_mass_range[1]),
        qpos_noise_low=qpos_noise_low,
        qpos_noise_high=qpos_noise_high,
        rally_count=int(kwargs.get("rally_count", 1)),
        reward_reach_dist_w=1.0,
        reward_palm_dist_w=1.0,
        reward_paddle_quat_w=2.0,
        reward_act_reg_w=0.5,
        reward_torso_up_w=2.0,
        reward_sparse_w=100.0,
        reward_solved_w=1000.0,
        reward_done_w=-10.0,
    )
    if overrides:
        config = MyoChallengeTableTennisEnvSpec.gen_config(
            **dict(
                zip(
                    MyoChallengeTableTennisEnvSpec._config_keys,
                    config,
                    strict=False,
                ),
                **overrides,
            )
        )
    return (
        config,
        MyoChallengeTableTennisGymnasiumEnvPool,
        MyoChallengeTableTennisEnvSpec,
    )


def _tabletennis_pixel_config(
    env_id: str,
) -> tuple[tuple[Any, ...], type, type]:
    config, _, _ = _tabletennis_config(env_id)
    values = dict(
        zip(MyoChallengeTableTennisEnvSpec._config_keys, config, strict=False)
    )
    pixel = MyoChallengeTableTennisPixelEnvSpec.gen_config(
        **values,
        render_width=64,
        render_height=64,
        render_camera_id=-1,
    )
    return (
        pixel,
        MyoChallengeTableTennisPixelGymnasiumEnvPool,
        MyoChallengeTableTennisPixelEnvSpec,
    )


def _oracle_class(env_id: str) -> Any:
    entry = _entry(env_id)
    return load_oracle_class(entry["entry_module"], entry["class_name"])


def _oracle_kwargs(env_id: str) -> dict[str, Any]:
    kwargs = dict(_entry(env_id)["kwargs"])
    for path_key in ("model_path", "init_pose_path"):
        if path_key in kwargs:
            kwargs[path_key] = str(_oracle_model_path(kwargs[path_key]))
    return kwargs


def _oracle_live_obs(unwrapped: Any) -> np.ndarray:
    obs_dict = unwrapped.get_obs_dict(unwrapped.sim)
    _, obs = unwrapped.obsdict2obsvec(obs_dict, unwrapped.obs_keys)
    return np.asarray(obs, dtype=np.float64)


def _oracle_reset_sync(
    env: Any, env_id: str
) -> tuple[np.ndarray, dict[str, Any]]:
    obs, _ = env.reset()
    unwrapped = env.unwrapped
    sim = unwrapped.sim
    sync: dict[str, Any] = {
        "test_reset_qpos": sim.data.qpos.copy().tolist(),
        "test_reset_qvel": sim.data.qvel.copy().tolist(),
        "test_reset_act": (
            sim.data.act.copy().tolist() if sim.model.na > 0 else []
        ),
        "test_reset_qacc_warmstart": sim.data.qacc_warmstart.copy().tolist(),
    }
    entry = _entry(env_id)
    if entry["class_name"] == "ReorientEnvV0":
        sync["test_reset_act_dot"] = (
            sim.data.act_dot.copy().tolist() if sim.model.na > 0 else []
        )
        goal_bid = sim.model.body_name2id("target")
        target_gid = sim.model.geom_name2id("target_dice")
        object_bid = sim.model.body_name2id("Object")
        start_geom = sim.model.body_geomadr[object_bid]
        geom_count = sim.model.body_geomnum[object_bid]
        sync["test_goal_body_pos"] = (
            sim.model.body_pos[goal_bid].copy().tolist()
        )
        sync["test_goal_body_quat"] = (
            sim.model.body_quat[goal_bid].copy().tolist()
        )
        sync["test_target_geom_size"] = (
            sim.model.geom_size[target_gid].copy().tolist()
        )
        sync["test_object_geom_size"] = (
            sim.model
            .geom_size[start_geom : start_geom + geom_count]
            .reshape(-1)
            .tolist()
        )
        sync["test_object_geom_pos"] = (
            sim.model
            .geom_pos[start_geom : start_geom + geom_count]
            .reshape(-1)
            .tolist()
        )
        sync["test_object_geom_friction"] = (
            sim.model
            .geom_friction[start_geom : start_geom + geom_count]
            .reshape(-1)
            .tolist()
        )
        sync["test_object_body_mass"] = [float(sim.model.body_mass[object_bid])]
    elif entry["class_name"] == "RelocateEnvV0":
        sync["test_reset_act_dot"] = (
            sim.data.act_dot.copy().tolist() if sim.model.na > 0 else []
        )
        sync["test_reset_ctrl"] = sim.data.ctrl.copy().tolist()
        goal_bid = sim.model.body_name2id("target")
        goal_mocap_id = int(sim.model.body_mocapid[goal_bid])
        object_bid = sim.model.body_name2id("Object")
        start_geom = sim.model.body_geomadr[object_bid]
        geom_count = sim.model.body_geomnum[object_bid]
        sync["test_goal_body_pos"] = (
            sim.model.body_pos[goal_bid].copy().tolist()
        )
        sync["test_goal_body_quat"] = (
            sim.model.body_quat[goal_bid].copy().tolist()
        )
        if goal_mocap_id >= 0:
            sync["test_goal_mocap_pos"] = (
                sim.data.mocap_pos[goal_mocap_id].copy().tolist()
            )
            sync["test_goal_mocap_quat"] = (
                sim.data.mocap_quat[goal_mocap_id].copy().tolist()
            )
        sync["test_object_body_pos"] = (
            sim.model.body_pos[object_bid].copy().tolist()
        )
        sync["test_object_body_mass"] = [float(sim.model.body_mass[object_bid])]
        sync["test_object_geom_type"] = (
            sim.model
            .geom_type[start_geom : start_geom + geom_count]
            .astype(np.float64)
            .tolist()
        )
        sync["test_object_geom_size"] = (
            sim.model
            .geom_size[start_geom : start_geom + geom_count]
            .reshape(-1)
            .tolist()
        )
        sync["test_object_geom_pos"] = (
            sim.model
            .geom_pos[start_geom : start_geom + geom_count]
            .reshape(-1)
            .tolist()
        )
        sync["test_object_geom_quat"] = (
            sim.model
            .geom_quat[start_geom : start_geom + geom_count]
            .reshape(-1)
            .tolist()
        )
        sync["test_object_geom_rgba"] = (
            sim.model
            .geom_rgba[start_geom : start_geom + geom_count]
            .reshape(-1)
            .tolist()
        )
        sync["test_object_geom_friction"] = (
            sim.model
            .geom_friction[start_geom : start_geom + geom_count]
            .reshape(-1)
            .tolist()
        )
        # Upstream can recursively re-reset here to avoid contact, but still
        # return the first stale observation. Align against the final live sim.
        obs = _oracle_live_obs(unwrapped)
    elif entry["class_name"] == "BaodingEnvV1":
        sync["test_task"] = int(unwrapped.which_task.value)
        sync["test_ball1_starting_angle"] = float(
            unwrapped.ball_1_starting_angle
        )
        sync["test_ball2_starting_angle"] = float(
            unwrapped.ball_2_starting_angle
        )
        sync["test_x_radius"] = float(unwrapped.x_radius)
        sync["test_y_radius"] = float(unwrapped.y_radius)
        sync["test_goal_trajectory"] = (
            np.asarray(unwrapped.goal, dtype=np.float64).reshape(-1).tolist()
        )
        sync["test_target1_site_pos"] = (
            sim.model.site_pos[unwrapped.target1_sid].copy().tolist()
        )
        sync["test_target2_site_pos"] = (
            sim.model.site_pos[unwrapped.target2_sid].copy().tolist()
        )
        sync["test_object1_body_mass"] = [
            float(sim.model.body_mass[unwrapped.object1_bid])
        ]
        sync["test_object2_body_mass"] = [
            float(sim.model.body_mass[unwrapped.object2_bid])
        ]
        sync["test_object1_geom_size"] = (
            sim.model.geom_size[unwrapped.object1_gid].copy().tolist()
        )
        sync["test_object2_geom_size"] = (
            sim.model.geom_size[unwrapped.object2_gid].copy().tolist()
        )
        sync["test_object1_geom_friction"] = (
            sim.model.geom_friction[unwrapped.object1_gid].copy().tolist()
        )
        sync["test_object2_geom_friction"] = (
            sim.model.geom_friction[unwrapped.object2_gid].copy().tolist()
        )
    elif entry["class_name"] == "BimanualEnvV1":
        sync["test_start_pos"] = np.asarray(
            unwrapped.start_pos, dtype=np.float64
        ).tolist()
        sync["test_goal_pos"] = np.asarray(
            unwrapped.goal_pos, dtype=np.float64
        ).tolist()
        sync["test_object_body_mass"] = [
            float(sim.model.body_mass[unwrapped.obj_bid])
        ]
        sync["test_object_geom_size"] = (
            sim.model.geom_size[unwrapped.obj_gid].copy().tolist()
        )
        sync["test_object_geom_friction"] = (
            sim.model.geom_friction[unwrapped.obj_gid].copy().tolist()
        )
        obs = np.asarray(unwrapped.get_obs(), dtype=np.float64)
    elif entry["class_name"] == "RunTrack":
        terrain_geom_id = sim.model.geom_name2id("terrain")
        hfield_id = int(sim.model.geom_dataid[terrain_geom_id])
        nrow = int(sim.model.hfield_nrow[hfield_id])
        ncol = int(sim.model.hfield_ncol[hfield_id])
        adr = int(sim.model.hfield_adr[hfield_id])
        sync["test_hfield_data"] = (
            sim.model.hfield_data[adr : adr + nrow * ncol].copy().tolist()
        )
        sync["test_terrain_geom_rgba"] = (
            sim.model.geom_rgba[terrain_geom_id].copy().tolist()
        )
        sync["test_terrain_geom_pos"] = (
            sim.model.geom_pos[terrain_geom_id].copy().tolist()
        )
        sync["test_terrain_type"] = int(
            np.asarray(unwrapped.terrain_type).reshape(-1)[0]
        )
        sync["test_osl_state"] = _runtrack_osl_state_id(
            unwrapped.OSL_CTRL.STATE_MACHINE.current_state.get_name()
        )
        obs = np.asarray(unwrapped.get_obs(), dtype=np.float64)
    elif entry["class_name"] == "SoccerEnvV0":
        sync["test_reset_ctrl"] = (
            sim.data.ctrl.copy().tolist() if sim.model.nu > 0 else []
        )
        sync["test_reset_act_dot"] = (
            sim.data.act_dot.copy().tolist() if sim.model.na > 0 else []
        )
        sync["test_reset_qacc"] = sim.data.qacc.copy().tolist()
        goalkeeper = unwrapped.goalkeeper
        sync["test_goalkeeper_pose"] = np.asarray(
            goalkeeper.get_goalkeeper_pose(), dtype=np.float64
        ).tolist()
        sync["test_goalkeeper_velocity"] = np.asarray(
            goalkeeper.goalkeeper_vel, dtype=np.float64
        ).tolist()
        sync["test_goalkeeper_noise_buffer"] = (
            np
            .asarray(goalkeeper.noise_process.buffer, dtype=np.float64)
            .reshape(-1)
            .tolist()
        )
        sync["test_goalkeeper_noise_idx"] = int(goalkeeper.noise_process.idx)
        sync["test_goalkeeper_block_velocity"] = float(
            goalkeeper.block_velocity
        )
        sync["test_goalkeeper_policy"] = _soccer_policy_id(
            goalkeeper.goalkeeper_policy
        )
        # Soccer mutates `sim` after reset and only forwards the live sim, not
        # the observed sim used by `get_obs()`. Align against the final live
        # sim state instead of the stale reset return value.
        obs = _oracle_live_obs(unwrapped)
    elif entry["class_name"] == "ChaseTagEnvV0":
        sync["test_reset_ctrl"] = (
            sim.data.ctrl.copy().tolist() if sim.model.nu > 0 else []
        )
        sync["test_reset_act_dot"] = (
            sim.data.act_dot.copy().tolist() if sim.model.na > 0 else []
        )
        sync["test_reset_qacc"] = sim.data.qacc.copy().tolist()
        opponent = unwrapped.opponent
        terrain_geom_id = sim.model.geom_name2id("terrain")
        hfield_id = int(sim.model.geom_dataid[terrain_geom_id])
        sync["test_task"] = int(unwrapped.current_task.value)
        sync["test_hfield_data"] = (
            sim.model
            .hfield_data[
                int(sim.model.hfield_adr[hfield_id]) : int(
                    sim.model.hfield_adr[hfield_id]
                )
                + int(sim.model.hfield_nrow[hfield_id])
                * int(sim.model.hfield_ncol[hfield_id])
            ]
            .copy()
            .tolist()
        )
        sync["test_terrain_geom_rgba"] = (
            sim.model.geom_rgba[terrain_geom_id].copy().tolist()
        )
        sync["test_terrain_geom_pos"] = (
            sim.model.geom_pos[terrain_geom_id].copy().tolist()
        )
        sync["test_opponent_pose"] = np.asarray(
            opponent.get_opponent_pose(), dtype=np.float64
        ).tolist()
        sync["test_opponent_velocity"] = np.asarray(
            opponent.opponent_vel, dtype=np.float64
        ).tolist()
        sync["test_opponent_noise_buffer"] = (
            np
            .asarray(opponent.noise_process.buffer, dtype=np.float64)
            .reshape(-1)
            .tolist()
        )
        sync["test_opponent_noise_idx"] = int(opponent.noise_process.idx)
        sync["test_chase_velocity"] = float(opponent.chase_velocity)
        sync["test_opponent_policy"] = _chasetag_policy_id(
            opponent.opponent_policy
        )
        # ChaseTag has the same live-sim vs observed-sim reset split as Soccer.
        obs = _oracle_live_obs(unwrapped)
    elif entry["class_name"] == "TableTennisEnvV0":
        sync["test_ball_body_pos"] = (
            sim.model.body_pos[unwrapped.id_info.ball_bid].copy().tolist()
        )
        sync["test_ball_geom_friction"] = (
            sim.model.geom_friction[unwrapped.id_info.ball_gid].copy().tolist()
        )
        sync["test_paddle_body_mass"] = [
            float(sim.model.body_mass[unwrapped.id_info.paddle_bid])
        ]
        sync["test_init_qpos"] = np.asarray(
            unwrapped.init_qpos, dtype=np.float64
        ).tolist()
        sync["test_init_qvel"] = np.asarray(
            unwrapped.init_qvel, dtype=np.float64
        ).tolist()
        sync["test_reset_act_dot"] = (
            sim.data.act_dot.copy().tolist() if sim.model.na > 0 else []
        )
    return obs, sync


def _alignment_obs_atol(env_id: str) -> float:
    class_name = _entry(env_id)["class_name"]
    if class_name == "BimanualEnvV1":
        return _BIMANUAL_ALIGNMENT_ATOL
    if class_name in {"RunTrack", "SoccerEnvV0", "ChaseTagEnvV0"}:
        return _LOCOMOTION_ALIGNMENT_ATOL
    if class_name == "TableTennisEnvV0":
        return _TABLETENNIS_ALIGNMENT_ATOL
    return _DEFAULT_ALIGNMENT_ATOL


def _alignment_reward_atol(env_id: str) -> float:
    class_name = _entry(env_id)["class_name"]
    if class_name in {"RunTrack", "SoccerEnvV0", "ChaseTagEnvV0"}:
        return _LOCOMOTION_ALIGNMENT_ATOL
    if class_name in {"BimanualEnvV1", "TableTennisEnvV0"}:
        return _TABLETENNIS_ALIGNMENT_ATOL
    return _DEFAULT_ALIGNMENT_ATOL


def _assert_alignment_with_oracle(
    case: absltest.TestCase,
    env_id: str,
    config_fn: Any,
    *,
    action_seed: int,
    steps: int = _ALIGNMENT_STEPS,
) -> None:
    entry = _entry(env_id)
    oracle_cls = _oracle_class(env_id)
    oracle: Any = gymnasium.wrappers.TimeLimit(
        oracle_cls(seed=42, **_oracle_kwargs(env_id)),
        max_episode_steps=int(entry["max_episode_steps"]),
    )
    obs0, sync = _oracle_reset_sync(oracle, env_id)
    config, pool_type, spec_type = config_fn(env_id, overrides=sync)
    native = _make_env(config, pool_type, spec_type)
    obs_atol = _alignment_obs_atol(env_id)
    reward_atol = _alignment_reward_atol(env_id)
    try:
        obs1, _ = native.reset()
        np.testing.assert_allclose(obs1[0], obs0, atol=obs_atol, rtol=1e-5)
        actions = _seeded_actions(
            _batched_action_shape(native), steps, action_seed
        )
        for action in actions:
            obs0, reward0, terminated0, truncated0, _ = oracle.step(action[0])
            obs1, reward1, terminated1, truncated1, _ = native.step(action)
            np.testing.assert_allclose(obs1[0], obs0, atol=obs_atol, rtol=1e-5)
            np.testing.assert_allclose(
                reward1[0], reward0, atol=reward_atol, rtol=1e-5
            )
            case.assertEqual(bool(terminated1[0]), bool(terminated0))
            case.assertEqual(bool(truncated1[0]), bool(truncated0))
            if terminated0 or truncated0:
                break
    finally:
        native.close()
        oracle.close()


class MyoSuiteMyoChallengeNativeTest(absltest.TestCase):
    """Covers the internal native MyoChallenge slice implemented so far."""

    def test_reorient_construct_and_reset_all_ids(self) -> None:
        """Every native reorient ID should construct and reset."""
        for env_id in _REORIENT_IDS:
            with self.subTest(env_id=env_id):
                config, pool_type, spec_type = _reorient_config(env_id)
                env = _make_env(config, pool_type, spec_type)
                obs, _ = env.reset()
                self.assertEqual(obs.shape[0], 1)
                env.close()

    def test_relocate_construct_and_reset_all_ids(self) -> None:
        """Every native relocate ID should construct and reset."""
        for env_id in _RELOCATE_IDS:
            with self.subTest(env_id=env_id):
                config, pool_type, spec_type = _relocate_config(env_id)
                env = _make_env(config, pool_type, spec_type)
                obs, _ = env.reset()
                self.assertEqual(obs.shape[0], 1)
                env.close()

    def test_reorient_native_determinism(self) -> None:
        """Reorient rollouts should be deterministic for a fixed action trace."""
        for env_id in _REORIENT_IDS:
            with self.subTest(env_id=env_id):
                config, pool_type, spec_type = _reorient_config(env_id)
                env0 = _make_env(config, pool_type, spec_type)
                env1 = _make_env(config, pool_type, spec_type)
                actions = _seeded_actions(_batched_action_shape(env0), 12, 1234)
                _assert_rollouts_match(
                    self, env0, env1, actions, atol=1e-8, rtol=1e-8
                )
                env0.close()
                env1.close()

    def test_relocate_native_determinism(self) -> None:
        """Relocate rollouts should be deterministic for a fixed action trace."""
        for env_id in _RELOCATE_IDS:
            with self.subTest(env_id=env_id):
                config, pool_type, spec_type = _relocate_config(env_id)
                env0 = _make_env(config, pool_type, spec_type)
                env1 = _make_env(config, pool_type, spec_type)
                actions = _seeded_actions(_batched_action_shape(env0), 12, 5678)
                _assert_rollouts_match(
                    self, env0, env1, actions, atol=1e-8, rtol=1e-8
                )
                env0.close()
                env1.close()

    def test_baoding_construct_and_reset_all_ids(self) -> None:
        """Every native baoding ID should construct and reset."""
        for env_id in _BAODING_IDS:
            with self.subTest(env_id=env_id):
                config, pool_type, spec_type = _baoding_config(env_id)
                env = _make_env(config, pool_type, spec_type)
                obs, _ = env.reset()
                self.assertEqual(obs.shape[0], 1)
                env.close()

    def test_baoding_native_determinism(self) -> None:
        """Baoding rollouts should be deterministic for a fixed action trace."""
        for env_id in _BAODING_IDS:
            with self.subTest(env_id=env_id):
                config, pool_type, spec_type = _baoding_config(env_id)
                env0 = _make_env(config, pool_type, spec_type)
                env1 = _make_env(config, pool_type, spec_type)
                actions = _seeded_actions(_batched_action_shape(env0), 12, 2468)
                _assert_rollouts_match(
                    self, env0, env1, actions, atol=1e-8, rtol=1e-8
                )
                env0.close()
                env1.close()

    def test_bimanual_construct_and_reset_all_ids(self) -> None:
        """Every native bimanual ID should construct and reset."""
        for env_id in _BIMANUAL_IDS:
            with self.subTest(env_id=env_id):
                config, pool_type, spec_type = _bimanual_config(env_id)
                env = _make_env(config, pool_type, spec_type)
                obs, _ = env.reset()
                self.assertEqual(obs.shape[0], 1)
                env.close()

    def test_bimanual_native_determinism(self) -> None:
        """Bimanual rollouts should be deterministic for a fixed action trace."""
        for env_id in _BIMANUAL_IDS:
            with self.subTest(env_id=env_id):
                config, pool_type, spec_type = _bimanual_config(env_id)
                env0 = _make_env(config, pool_type, spec_type)
                env1 = _make_env(config, pool_type, spec_type)
                actions = _seeded_actions(_batched_action_shape(env0), 12, 9753)
                _assert_rollouts_match(
                    self, env0, env1, actions, atol=1e-8, rtol=1e-8
                )
                env0.close()
                env1.close()

    def test_runtrack_construct_and_reset_all_ids(self) -> None:
        """Every native RunTrack ID should construct and reset."""
        for env_id in _RUNTRACK_IDS:
            with self.subTest(env_id=env_id):
                config, pool_type, spec_type = _runtrack_config(env_id)
                env = _make_env(config, pool_type, spec_type)
                obs, _ = env.reset()
                self.assertEqual(obs.shape[0], 1)
                env.close()

    def test_runtrack_native_determinism(self) -> None:
        """RunTrack rollouts should be deterministic for a fixed action trace."""
        for env_id in _RUNTRACK_IDS:
            with self.subTest(env_id=env_id):
                config, pool_type, spec_type = _runtrack_config(env_id)
                env0 = _make_env(config, pool_type, spec_type)
                env1 = _make_env(config, pool_type, spec_type)
                actions = _seeded_actions(_batched_action_shape(env0), 12, 1597)
                _assert_rollouts_match(
                    self, env0, env1, actions, atol=1e-8, rtol=1e-8
                )
                env0.close()
                env1.close()

    def test_soccer_construct_and_reset_all_ids(self) -> None:
        """Every native soccer ID should construct and reset."""
        for env_id in _SOCCER_IDS:
            with self.subTest(env_id=env_id):
                config, pool_type, spec_type = _soccer_config(env_id)
                env = _make_env(config, pool_type, spec_type)
                obs, _ = env.reset()
                self.assertEqual(obs.shape[0], 1)
                env.close()

    def test_soccer_native_determinism(self) -> None:
        """Soccer rollouts should be deterministic for a fixed action trace."""
        for env_id in _SOCCER_IDS:
            with self.subTest(env_id=env_id):
                config, pool_type, spec_type = _soccer_config(env_id)
                env0 = _make_env(config, pool_type, spec_type)
                env1 = _make_env(config, pool_type, spec_type)
                actions = _seeded_actions(_batched_action_shape(env0), 12, 1601)
                _assert_rollouts_match(
                    self, env0, env1, actions, atol=1e-8, rtol=1e-8
                )
                env0.close()
                env1.close()

    def test_chasetag_construct_and_reset_all_ids(self) -> None:
        """Every native ChaseTag ID should construct and reset."""
        for env_id in _CHASETAG_IDS:
            with self.subTest(env_id=env_id):
                config, pool_type, spec_type = _chasetag_config(env_id)
                env = _make_env(config, pool_type, spec_type)
                obs, _ = env.reset()
                self.assertEqual(obs.shape[0], 1)
                env.close()

    def test_chasetag_native_determinism(self) -> None:
        """ChaseTag rollouts should be deterministic for a fixed action trace."""
        for env_id in _CHASETAG_IDS:
            with self.subTest(env_id=env_id):
                config, pool_type, spec_type = _chasetag_config(env_id)
                env0 = _make_env(config, pool_type, spec_type)
                env1 = _make_env(config, pool_type, spec_type)
                actions = _seeded_actions(_batched_action_shape(env0), 12, 1607)
                _assert_rollouts_match(
                    self, env0, env1, actions, atol=1e-8, rtol=1e-8
                )
                env0.close()
                env1.close()

    def test_tabletennis_construct_and_reset_all_ids(self) -> None:
        """Every native TableTennis ID should construct and reset."""
        for env_id in _TABLETENNIS_IDS:
            with self.subTest(env_id=env_id):
                config, pool_type, spec_type = _tabletennis_config(env_id)
                env = _make_env(config, pool_type, spec_type)
                obs, _ = env.reset()
                self.assertEqual(obs.shape[0], 1)
                env.close()

    def test_tabletennis_native_determinism(self) -> None:
        """TableTennis rollouts should be deterministic for a fixed action trace."""
        for env_id in _TABLETENNIS_IDS:
            with self.subTest(env_id=env_id):
                config, pool_type, spec_type = _tabletennis_config(env_id)
                env0 = _make_env(config, pool_type, spec_type)
                env1 = _make_env(config, pool_type, spec_type)
                actions = _seeded_actions(_batched_action_shape(env0), 12, 1613)
                _assert_rollouts_match(
                    self, env0, env1, actions, atol=1e-8, rtol=1e-8
                )
                env0.close()
                env1.close()

    def test_reorient_alignment_with_oracle(self) -> None:
        """Reorient should align stepwise with the official oracle."""
        cls = _oracle_class(_REORIENT_IDS[0])
        for env_id in _REORIENT_IDS:
            with self.subTest(env_id=env_id):
                oracle: Any = gymnasium.wrappers.TimeLimit(
                    cls(seed=123, **_oracle_kwargs(env_id)),
                    max_episode_steps=int(_entry(env_id)["max_episode_steps"]),
                )
                obs0, sync = _oracle_reset_sync(oracle, env_id)
                config, pool_type, spec_type = _reorient_config(
                    env_id, overrides=sync
                )
                native = _make_env(config, pool_type, spec_type)
                obs1, _ = native.reset()
                _assert_reorient_obs_close(obs1[0], obs0)
                actions = _seeded_actions(
                    _batched_action_shape(native), _ALIGNMENT_STEPS, 4321
                )
                for action in actions:
                    obs0, reward0, terminated0, truncated0, _ = oracle.step(
                        action[0]
                    )
                    obs1, reward1, terminated1, truncated1, _ = native.step(
                        action
                    )
                    _assert_reorient_obs_close(obs1[0], obs0)
                    np.testing.assert_allclose(
                        reward1[0],
                        float(reward0),
                        atol=_reorient_alignment_reward_atol(),
                        rtol=1e-5,
                    )
                    self.assertEqual(bool(terminated1[0]), bool(terminated0))
                    self.assertEqual(bool(truncated1[0]), bool(truncated0))
                    if terminated0 or truncated0:
                        break
                native.close()
                oracle.close()

    def test_reorient_fatigue_first_step_act_matches_oracle(self) -> None:
        """Fatigue Reorient should preserve upstream warm-started act state."""
        env_id = "myoFatiChallengeDieReorientP1-v0"
        oracle: Any = gymnasium.wrappers.TimeLimit(
            _oracle_class(env_id)(seed=123, **_oracle_kwargs(env_id)),
            max_episode_steps=int(_entry(env_id)["max_episode_steps"]),
        )
        obs0, sync = _oracle_reset_sync(oracle, env_id)
        config, pool_type, spec_type = _reorient_config(env_id, overrides=sync)
        native = _make_env(config, pool_type, spec_type)
        obs1, _ = native.reset()
        _assert_reorient_obs_close(obs1[0], obs0)
        action = _seeded_actions(_batched_action_shape(native), 1, 104)[0]
        obs0, reward0, terminated0, truncated0, _ = oracle.step(action[0])
        obs1, reward1, terminated1, truncated1, _ = native.step(action)
        act_dim = oracle.unwrapped.sim.model.na
        np.testing.assert_allclose(
            obs1[0][-act_dim:],
            obs0[-act_dim:],
            atol=5e-8,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            reward1[0], float(reward0), atol=5e-8, rtol=1e-8
        )
        self.assertEqual(bool(terminated1[0]), bool(terminated0))
        self.assertEqual(bool(truncated1[0]), bool(truncated0))
        native.close()
        oracle.close()

    def test_relocate_alignment_with_oracle(self) -> None:
        """Relocate should align stepwise with the official oracle."""
        cls = _oracle_class(_RELOCATE_IDS[0])
        for env_id in _RELOCATE_IDS:
            with self.subTest(env_id=env_id):
                oracle: Any = gymnasium.wrappers.TimeLimit(
                    cls(seed=123, **_oracle_kwargs(env_id)),
                    max_episode_steps=int(_entry(env_id)["max_episode_steps"]),
                )
                obs0, sync = _oracle_reset_sync(oracle, env_id)
                config, pool_type, spec_type = _relocate_config(
                    env_id, overrides=sync
                )
                native = _make_env(config, pool_type, spec_type)
                obs1, _ = native.reset()
                # Official relocate observations are reconstructed via robot
                # sensor->sim propagation, which leaves an isolated ~2e-4 drift
                # on object_z / pos_err_z after reset-sync.
                np.testing.assert_allclose(
                    obs1[0],
                    obs0,
                    atol=_relocate_alignment_obs_atol(env_id),
                    rtol=1e-5,
                )
                actions = _seeded_actions(
                    _batched_action_shape(native), _ALIGNMENT_STEPS, 8765
                )
                for action in actions:
                    obs0, reward0, terminated0, truncated0, _ = oracle.step(
                        action[0]
                    )
                    obs1, reward1, terminated1, truncated1, _ = native.step(
                        action
                    )
                    np.testing.assert_allclose(
                        obs1[0],
                        obs0,
                        atol=_relocate_alignment_obs_atol(env_id),
                        rtol=1e-5,
                    )
                    np.testing.assert_allclose(
                        reward1[0],
                        float(reward0),
                        atol=_relocate_alignment_reward_atol(env_id),
                        rtol=1e-5,
                    )
                    self.assertEqual(bool(terminated1[0]), bool(terminated0))
                    self.assertEqual(bool(truncated1[0]), bool(truncated0))
                    if terminated0 or truncated0:
                        break
                native.close()
                oracle.close()

    def test_baoding_alignment_with_oracle(self) -> None:
        """Baoding should align stepwise with the official oracle."""
        for env_id in _BAODING_IDS:
            with self.subTest(env_id=env_id):
                _assert_alignment_with_oracle(
                    self, env_id, _baoding_config, action_seed=1123
                )

    def test_bimanual_alignment_with_oracle(self) -> None:
        """Bimanual should align stepwise with the official oracle."""
        for env_id in _BIMANUAL_IDS:
            with self.subTest(env_id=env_id):
                _assert_alignment_with_oracle(
                    self, env_id, _bimanual_config, action_seed=1223
                )

    def test_runtrack_alignment_with_oracle(self) -> None:
        """RunTrack should align stepwise with the official oracle."""
        for env_id in _RUNTRACK_IDS:
            with self.subTest(env_id=env_id):
                _assert_alignment_with_oracle(
                    self,
                    env_id,
                    _runtrack_config,
                    action_seed=1323,
                )

    def test_soccer_alignment_with_oracle(self) -> None:
        """Soccer should align stepwise with the official oracle."""
        for env_id in _SOCCER_IDS:
            with self.subTest(env_id=env_id):
                _assert_alignment_with_oracle(
                    self,
                    env_id,
                    _soccer_config,
                    action_seed=1423,
                )

    def test_chasetag_alignment_with_oracle(self) -> None:
        """ChaseTag should align stepwise with the official oracle."""
        for env_id in _CHASETAG_IDS:
            with self.subTest(env_id=env_id):
                _assert_alignment_with_oracle(
                    self,
                    env_id,
                    _chasetag_config,
                    action_seed=1523,
                )

    def test_tabletennis_alignment_with_oracle(self) -> None:
        """TableTennis should align stepwise with the official oracle."""
        for env_id in _TABLETENNIS_IDS:
            with self.subTest(env_id=env_id):
                _assert_alignment_with_oracle(
                    self,
                    env_id,
                    _tabletennis_config,
                    action_seed=1623,
                )

    def test_reorient_pixel_observation_smoke(self) -> None:
        """Reorient pixel wrappers should emit non-empty batched frames."""
        config, pool_type, spec_type = _reorient_pixel_config(_REORIENT_IDS[0])
        env = _make_env(config, pool_type, spec_type)
        obs, _ = env.reset()
        self.assertEqual(obs.shape[0], 1)
        self.assertGreater(obs.size, 0)
        env.close()

    def test_relocate_pixel_observation_smoke(self) -> None:
        """Relocate pixel wrappers should emit non-empty batched frames."""
        config, pool_type, spec_type = _relocate_pixel_config(_RELOCATE_IDS[0])
        env = _make_env(config, pool_type, spec_type)
        obs, _ = env.reset()
        self.assertEqual(obs.shape[0], 1)
        self.assertGreater(obs.size, 0)
        env.close()

    def test_baoding_pixel_observation_smoke(self) -> None:
        """Baoding pixel wrappers should emit non-empty batched frames."""
        config, pool_type, spec_type = _baoding_pixel_config(_BAODING_IDS[0])
        env = _make_env(config, pool_type, spec_type)
        obs, _ = env.reset()
        self.assertEqual(obs.shape[0], 1)
        self.assertGreater(obs.size, 0)
        env.close()

    def test_bimanual_pixel_observation_smoke(self) -> None:
        """Bimanual pixel wrappers should emit non-empty batched frames."""
        config, pool_type, spec_type = _bimanual_pixel_config(_BIMANUAL_IDS[0])
        env = _make_env(config, pool_type, spec_type)
        obs, _ = env.reset()
        self.assertEqual(obs.shape[0], 1)
        self.assertGreater(obs.size, 0)
        env.close()

    def test_runtrack_pixel_observation_smoke(self) -> None:
        """RunTrack pixel wrappers should emit non-empty batched frames."""
        config, pool_type, spec_type = _runtrack_pixel_config(_RUNTRACK_IDS[0])
        env = _make_env(config, pool_type, spec_type)
        obs, _ = env.reset()
        self.assertEqual(obs.shape[0], 1)
        self.assertGreater(obs.size, 0)
        env.close()

    def test_soccer_pixel_observation_smoke(self) -> None:
        """Soccer pixel wrappers should emit non-empty batched frames."""
        config, pool_type, spec_type = _soccer_pixel_config(_SOCCER_IDS[0])
        env = _make_env(config, pool_type, spec_type)
        obs, _ = env.reset()
        self.assertEqual(obs.shape[0], 1)
        self.assertGreater(obs.size, 0)
        env.close()

    def test_chasetag_pixel_observation_smoke(self) -> None:
        """ChaseTag pixel wrappers should emit non-empty batched frames."""
        config, pool_type, spec_type = _chasetag_pixel_config(_CHASETAG_IDS[0])
        env = _make_env(config, pool_type, spec_type)
        obs, _ = env.reset()
        self.assertEqual(obs.shape[0], 1)
        self.assertGreater(obs.size, 0)
        env.close()

    def test_tabletennis_pixel_observation_smoke(self) -> None:
        """TableTennis pixel wrappers should emit non-empty batched frames."""
        config, pool_type, spec_type = _tabletennis_pixel_config(
            _TABLETENNIS_IDS[0]
        )
        env = _make_env(config, pool_type, spec_type)
        obs, _ = env.reset()
        self.assertEqual(obs.shape[0], 1)
        self.assertGreater(obs.size, 0)
        env.close()


if __name__ == "__main__":
    absltest.main()
