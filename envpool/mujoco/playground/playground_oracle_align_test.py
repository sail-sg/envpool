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
"""Oracle alignment tests for MuJoCo Playground Go1."""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
import unittest
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from envpool.mujoco.oracle import configure_mujoco_package_shared_lib
from envpool.python.glfw_context import preload_windows_gl_dlls

preload_windows_gl_dlls(strict=True)


def _configure_linux_mujoco_gl() -> None:
    if platform.system() != "Linux":
        return
    if os.environ.get("MUJOCO_GL"):
        if os.environ["MUJOCO_GL"] == "egl":
            os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        return
    for backend in ("egl", "osmesa"):
        env = dict(os.environ)
        env["MUJOCO_GL"] = backend
        if backend == "egl":
            env.setdefault("EGL_PLATFORM", "surfaceless")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import mujoco; "
                    "ctx = mujoco.GLContext(1, 1); "
                    "ctx.make_current(); "
                    "ctx.free()"
                ),
            ],
            env=env,
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            os.environ["MUJOCO_GL"] = backend
            if backend == "egl":
                os.environ.setdefault("EGL_PLATFORM", "surfaceless")
            return


_configure_linux_mujoco_gl()
# Linux MjSpec bindings must retain their matching pip library: substituting
# another build crashes while compiling Barkour's contact sensors.
configure_mujoco_package_shared_lib()

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from absl.testing import absltest
from etils import epath
from mujoco import mjx
from mujoco_playground._src import locomotion as oracle_locomotion
from mujoco_playground._src import manipulation as oracle_manipulation
from mujoco_playground._src import mjx_env as oracle_mjx_env
from mujoco_playground._src.locomotion.apollo import (
    joystick as oracle_apollo,
)
from mujoco_playground._src.locomotion.barkour import (
    joystick as oracle_barkour,
)
from mujoco_playground._src.locomotion.berkeley_humanoid import (
    joystick as oracle_berkeley,
)
from mujoco_playground._src.locomotion.g1 import joystick as oracle_g1
from mujoco_playground._src.locomotion.go1 import getup as oracle_go1_getup
from mujoco_playground._src.locomotion.go1 import (
    handstand as oracle_go1_handstand,
)
from mujoco_playground._src.locomotion.go1 import joystick as oracle_go1
from mujoco_playground._src.locomotion.h1 import (
    inplace_gait_tracking as oracle_h1_inplace,
)
from mujoco_playground._src.locomotion.h1 import (
    joystick_gait_tracking as oracle_h1_joystick,
)
from mujoco_playground._src.locomotion.op3 import joystick as oracle_op3
from mujoco_playground._src.locomotion.spot import getup as oracle_spot_getup
from mujoco_playground._src.locomotion.spot import (
    joystick as oracle_spot_joystick,
)
from mujoco_playground._src.locomotion.spot import (
    joystick_gait_tracking as oracle_spot_gait,
)
from mujoco_playground._src.locomotion.t1 import joystick as oracle_t1
from mujoco_playground._src.manipulation.aero_hand import (
    rotate_z as oracle_aero_rotate,
)
from mujoco_playground._src.manipulation.aloha import (
    handover as oracle_aloha_handover,
)
from mujoco_playground._src.manipulation.aloha import (
    single_peg_insertion as oracle_aloha_peg,
)
from mujoco_playground._src.manipulation.franka_emika_panda import (
    open_cabinet as oracle_panda_open_cabinet,
)
from mujoco_playground._src.manipulation.franka_emika_panda import (
    pick as oracle_panda_pick,
)
from mujoco_playground._src.manipulation.franka_emika_panda import (
    pick_cartesian as oracle_panda_pick_cartesian,
)
from mujoco_playground._src.manipulation.franka_emika_panda_robotiq import (
    push_cube as oracle_panda_robotiq,
)
from mujoco_playground._src.manipulation.leap_hand import (
    reorient as oracle_leap_reorient,
)
from mujoco_playground._src.manipulation.leap_hand import (
    rotate_z as oracle_leap_rotate,
)

import envpool.mujoco.playground.registration as playground_registration
from envpool.mujoco.oracle import runfiles_repository
from envpool.mujoco.render_oracle import configure_macos_mujoco_renderer
from envpool.registration import make_gymnasium

configure_macos_mujoco_renderer()

_ALL_TASK_IDS = tuple(
    f"{name}-v1" for name in playground_registration.PLAYGROUND_ENVS
)
_TASK_FILTER = frozenset(
    task.strip()
    for task in os.environ.get("PLAYGROUND_TASK_FILTER", "").split(",")
    if task.strip()
)
_FILTERED_TASK_IDS = tuple(
    task_id
    for task_id in _ALL_TASK_IDS
    if not _TASK_FILTER or task_id in _TASK_FILTER
)
# The public Playground task is authored against MJX, but EnvPool's runtime is
# native MuJoCo C++. For the official Go1 XML, MJX and MuJoCo C do not match
# bitwise through the one-iteration frictionloss solver path; the diagnostic
# test below pins that gap and shows it disappears when frictionloss is disabled
# or when solver iterations are raised. The alignment oracle therefore uses the
# pinned upstream XML/assets and official obs/reward formulas while stepping the
# same MuJoCo C backend that EnvPool embeds.
_ROLLOUT_STEPS = 128
_RENDER_WIDTH = 64
_RENDER_HEIGHT = 48
_CONFIG = {
    "noise_level": 0.0,
}
_JOYSTICK_CONFIG = {
    **_CONFIG,
    "command_a0": 0.0,
    "command_a1": 0.0,
    "command_a2": 0.0,
    "command_b0": 0.0,
    "command_b1": 0.0,
    "command_b2": 0.0,
}
_APOLLO_CONFIG = {
    **_CONFIG,
    "push_enable": 0.0,
    "command_min0": 0.0,
    "command_min1": 0.0,
    "command_min2": 0.0,
    "command_max0": 0.0,
    "command_max1": 0.0,
    "command_max2": 0.0,
    "command_zero_prob0": 1.0,
    "command_zero_prob1": 1.0,
    "command_zero_prob2": 1.0,
}
_BARKOUR_CONFIG = {
    "obs_noise": -1.0,
    "lin_vel_x_min": 0.0,
    "lin_vel_x_max": 0.0,
    "lin_vel_y_min": 0.0,
    "lin_vel_y_max": 0.0,
    "ang_vel_yaw_min": 0.0,
    "ang_vel_yaw_max": 0.0,
    "kick_wait_steps_min": 100000,
    "kick_wait_steps_max": 100001,
}
_OP3_CONFIG = {
    "obs_noise": -1.0,
    "lin_vel_x_min": 0.0,
    "lin_vel_x_max": 0.0,
    "lin_vel_y_min": 0.0,
    "lin_vel_y_max": 0.0,
    "ang_vel_yaw_min": 0.0,
    "ang_vel_yaw_max": 0.0,
}
_BERKELEY_CONFIG = {
    "noise_level": 0.0,
    "push_enable": 0.0,
    "lin_vel_x_min": 0.0,
    "lin_vel_x_max": 0.0,
    "lin_vel_y_min": 0.0,
    "lin_vel_y_max": 0.0,
    "ang_vel_yaw_min": 0.0,
    "ang_vel_yaw_max": 0.0,
}
_G1_CONFIG = {
    "noise_level": 0.0,
    "push_enable": 0.0,
    "lin_vel_x_min": 0.0,
    "lin_vel_x_max": 0.0,
    "lin_vel_y_min": 0.0,
    "lin_vel_y_max": 0.0,
    "ang_vel_yaw_min": 0.0,
    "ang_vel_yaw_max": 0.0,
}
_T1_CONFIG = {
    "noise_level": 0.0,
    "push_enable": 0.0,
    "lin_vel_x_min": 0.0,
    "lin_vel_x_max": 0.0,
    "lin_vel_y_min": 0.0,
    "lin_vel_y_max": 0.0,
    "ang_vel_yaw_min": 0.0,
    "ang_vel_yaw_max": 0.0,
}
_H1_CONFIG = {
    "obs_noise_level": 0.0,
    "lin_vel_x_min": 0.0,
    "lin_vel_x_max": 0.0,
    "lin_vel_y_min": 0.0,
    "lin_vel_y_max": 0.0,
    "ang_vel_yaw_min": 0.0,
    "ang_vel_yaw_max": 0.0,
}
_SPOT_JOYSTICK_CONFIG = {
    "noise_level": 0.0,
    "pert_enable": 0.0,
    "lin_vel_x_min": 0.0,
    "lin_vel_x_max": 0.0,
    "lin_vel_y_min": 0.0,
    "lin_vel_y_max": 0.0,
    "ang_vel_yaw_min": 0.0,
    "ang_vel_yaw_max": 0.0,
}
_SPOT_GETUP_CONFIG = {
    "noise_level": 0.0,
}
_SPOT_GAIT_CONFIG = {
    "noise_level": 0.0,
    "lin_vel_x_min": 0.0,
    "lin_vel_x_max": 0.0,
    "lin_vel_y_min": 0.0,
    "lin_vel_y_max": 0.0,
    "ang_vel_yaw_min": 0.0,
    "ang_vel_yaw_max": 0.0,
}
_APOLLO_REWARD_INFO_KEYS = (
    "reward_termination",
    "reward_alive",
    "reward_tracking",
    "reward_lin_vel_z",
    "reward_ang_vel_xy",
    "reward_orientation",
    "reward_feet_phase",
    "reward_torques",
    "reward_action_rate",
    "reward_energy",
    "reward_collision",
    "reward_pose",
)
_BARKOUR_REWARD_INFO_KEYS = (
    "reward_tracking_lin_vel",
    "reward_tracking_ang_vel",
    "reward_lin_vel_z",
    "reward_ang_vel_xy",
    "reward_orientation",
    "reward_torques",
    "reward_action_rate",
    "reward_stand_still",
    "reward_termination",
    "reward_feet_air_time",
)
_OP3_REWARD_INFO_KEYS = (
    "reward_tracking_lin_vel",
    "reward_tracking_ang_vel",
    "reward_lin_vel_z",
    "reward_ang_vel_xy",
    "reward_orientation",
    "reward_torques",
    "reward_action_rate",
    "reward_zero_cmd",
    "reward_termination",
    "reward_feet_slip",
    "reward_feet_clearance",
    "reward_energy",
)
_BERKELEY_REWARD_INFO_KEYS = (
    "reward_tracking_lin_vel",
    "reward_tracking_ang_vel",
    "reward_lin_vel_z",
    "reward_ang_vel_xy",
    "reward_orientation",
    "reward_base_height",
    "reward_torques",
    "reward_action_rate",
    "reward_energy",
    "reward_feet_clearance",
    "reward_feet_air_time",
    "reward_feet_slip",
    "reward_feet_height",
    "reward_feet_phase",
    "reward_stand_still",
    "reward_alive",
    "reward_termination",
    "reward_joint_deviation_knee",
    "reward_joint_deviation_hip",
    "reward_dof_pos_limits",
    "reward_pose",
)
_G1_REWARD_INFO_KEYS = (
    "reward_tracking_lin_vel",
    "reward_tracking_ang_vel",
    "reward_lin_vel_z",
    "reward_ang_vel_xy",
    "reward_orientation",
    "reward_base_height",
    "reward_torques",
    "reward_action_rate",
    "reward_energy",
    "reward_dof_acc",
    "reward_feet_clearance",
    "reward_feet_air_time",
    "reward_feet_slip",
    "reward_feet_height",
    "reward_feet_phase",
    "reward_stand_still",
    "reward_alive",
    "reward_termination",
    "reward_collision",
    "reward_contact_force",
    "reward_joint_deviation_knee",
    "reward_joint_deviation_hip",
    "reward_dof_pos_limits",
    "reward_pose",
)
_T1_REWARD_INFO_KEYS = (
    "reward_tracking_lin_vel",
    "reward_tracking_ang_vel",
    "reward_lin_vel_z",
    "reward_ang_vel_xy",
    "reward_orientation",
    "reward_base_height",
    "reward_torques",
    "reward_action_rate",
    "reward_energy",
    "reward_dof_acc",
    "reward_dof_vel",
    "reward_feet_clearance",
    "reward_feet_air_time",
    "reward_feet_slip",
    "reward_feet_height",
    "reward_feet_phase",
    "reward_stand_still",
    "reward_alive",
    "reward_termination",
    "reward_collision",
    "reward_joint_deviation_knee",
    "reward_joint_deviation_hip",
    "reward_dof_pos_limits",
    "reward_pose",
    "reward_feet_distance",
)
_H1_INPLACE_REWARD_INFO_KEYS = (
    "reward_feet_phase",
    "reward_pose",
    "reward_ang_vel",
    "reward_lin_vel",
)
_H1_JOYSTICK_REWARD_INFO_KEYS = (
    "reward_feet_phase",
    "reward_tracking_lin_vel",
    "reward_tracking_ang_vel",
    "reward_feet_air_time",
    "reward_ang_vel_xy",
    "reward_lin_vel_z",
    "reward_pose",
    "reward_foot_slip",
    "reward_action_rate",
)
_SPOT_JOYSTICK_REWARD_INFO_KEYS = (
    "reward_tracking_lin_vel",
    "reward_tracking_ang_vel",
    "reward_lin_vel_z",
    "reward_ang_vel_xy",
    "reward_orientation",
    "reward_termination",
    "reward_posture",
    "reward_torques",
    "reward_action_rate",
    "reward_energy",
    "reward_feet_slip",
    "reward_feet_clearance",
    "reward_feet_height",
    "reward_feet_air_time",
)
_SPOT_GETUP_REWARD_INFO_KEYS = (
    "reward_orientation",
    "reward_torso_height",
    "reward_posture",
    "reward_stand_still",
    "reward_action_rate",
    "reward_torques",
)
_SPOT_GAIT_REWARD_INFO_KEYS = (
    "reward_tracking_lin_vel",
    "reward_tracking_ang_vel",
    "reward_feet_phase",
    "reward_ang_vel_xy",
    "reward_lin_vel_z",
    "reward_hip_splay",
)
_PANDA_PICK_REWARD_INFO_KEYS = (
    "reward_gripper_box",
    "reward_box_target",
    "reward_no_floor_collision",
    "reward_robot_target_qpos",
)
_PANDA_CARTESIAN_REWARD_INFO_KEYS = (
    "reward_gripper_box",
    "reward_box_target",
    "reward_no_floor_collision",
    "reward_no_box_collision",
    "reward_robot_target_qpos",
    "reward_lifted",
    "reward_success",
)
_PANDA_OPEN_CABINET_REWARD_INFO_KEYS = (
    "reward_gripper_box",
    "reward_box_target",
    "reward_no_barrier_collision",
    "reward_robot_target_qpos",
)
_PANDA_ROBOTIQ_REWARD_INFO_KEYS = (
    "reward_gripper_box",
    "reward_box_target",
    "reward_box_orientation",
    "reward_gripper_collision_side",
    "reward_robot_target_qpos",
    "reward_joint_vel",
    "reward_joint_vel_limit",
    "reward_total_command",
    "reward_action_rate",
)
_HAND_ROTATE_REWARD_INFO_KEYS = (
    "reward_angvel",
    "reward_linvel",
    "reward_pose",
    "reward_torques",
    "reward_energy",
    "reward_termination",
    "reward_action_rate",
)
_LEAP_REORIENT_REWARD_INFO_KEYS = (
    "reward_orientation",
    "reward_position",
    "reward_termination",
    "reward_hand_pose",
    "reward_action_rate",
    "reward_joint_vel",
    "reward_energy",
    "reward_success",
)
_ALOHA_HANDOVER_REWARD_INFO_KEYS = (
    "reward_gripper_box",
    "reward_box_handover",
    "reward_handover_target",
    "reward_no_table_collision",
)
_ALOHA_PEG_REWARD_INFO_KEYS = (
    "reward_left_reward",
    "reward_right_reward",
    "reward_left_target_qpos",
    "reward_right_target_qpos",
    "reward_no_table_collision",
    "reward_socket_z_up",
    "reward_peg_z_up",
    "reward_socket_entrance_reward",
    "reward_peg_end2_reward",
    "reward_peg_insertion_reward",
)
_REWARD_INFO_KEYS = (
    "reward_tracking_lin_vel",
    "reward_tracking_ang_vel",
    "reward_lin_vel_z",
    "reward_ang_vel_xy",
    "reward_orientation",
    "reward_dof_pos_limits",
    "reward_pose",
    "reward_termination",
    "reward_stand_still",
    "reward_torques",
    "reward_action_rate",
    "reward_energy",
    "reward_feet_clearance",
    "reward_feet_height",
    "reward_feet_slip",
    "reward_feet_air_time",
)
_GETUP_REWARD_INFO_KEYS = (
    "reward_orientation",
    "reward_torso_height",
    "reward_posture",
    "reward_stand_still",
    "reward_action_rate",
    "reward_dof_pos_limits",
    "reward_torques",
    "reward_dof_acc",
    "reward_dof_vel",
)
_HANDSTAND_REWARD_INFO_KEYS = (
    "reward_height",
    "reward_orientation",
    "reward_contact",
    "reward_action_rate",
    "reward_termination",
    "reward_dof_pos_limits",
    "reward_torques",
    "reward_pose",
    "reward_stay_still",
    "reward_energy",
    "reward_dof_acc",
)
_STRICT_MJX_DIAGNOSTIC_ATOL = 1e-10


def _bazel_shard() -> tuple[int, int]:
    if status_file := os.environ.get("TEST_SHARD_STATUS_FILE"):
        Path(status_file).touch()
    total = int(os.environ.get("TEST_TOTAL_SHARDS", "1"))
    index = int(os.environ.get("TEST_SHARD_INDEX", "0"))
    if total < 1 or not 0 <= index < total:
        return (0, 1)
    return (index, total)


def _shard_task_ids(task_ids: tuple[str, ...]) -> tuple[str, ...]:
    index, total = _bazel_shard()
    if total == 1:
        return task_ids
    return tuple(
        task_id
        for offset, task_id in enumerate(task_ids)
        if offset % total == index
    )


def _is_primary_shard() -> bool:
    index, _ = _bazel_shard()
    return index == 0


_TASK_IDS = _shard_task_ids(_FILTERED_TASK_IDS)


def _candidate_runfile_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("TEST_SRCDIR", "RUNFILES_DIR"):
        if os.environ.get(env_name):
            roots.append(Path(os.environ[env_name]))
    roots.append(Path.cwd())
    roots.extend(Path(__file__).resolve().parents)
    workspace = os.environ.get("TEST_WORKSPACE")
    if workspace:
        roots.extend(root / workspace for root in list(roots))
    return roots


@lru_cache
def _runfiles_manifest_entries() -> tuple[tuple[str, Path], ...]:
    manifest = os.environ.get("RUNFILES_MANIFEST_FILE")
    if not manifest:
        return ()
    entries = []
    with Path(manifest).open(encoding="utf-8") as f:
        for line in f:
            logical, sep, physical = line.rstrip("\n").partition(" ")
            entries.append((logical, Path(physical if sep else logical)))
    return tuple(entries)


def _find_manifest_runfile(relative: str) -> Path | None:
    relative = relative.replace("\\", "/")
    candidates = [relative]
    workspace = os.environ.get("TEST_WORKSPACE")
    if workspace:
        candidates.append(f"{workspace}/{relative}")
    for logical, physical in _runfiles_manifest_entries():
        for candidate in candidates:
            if logical == candidate:
                return physical
            prefix = f"{candidate.rstrip('/')}/"
            if logical.startswith(prefix):
                path = physical
                for _ in logical[len(prefix) :].split("/"):
                    path = path.parent
                return path
    return None


def _find_runfile(relative: str) -> Path:
    manifest_path = _find_manifest_runfile(relative)
    if manifest_path is not None:
        return manifest_path
    for root in _candidate_runfile_roots():
        path = root / relative
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find runfile {relative!r}")


def _configure_oracle_assets() -> None:
    oracle_mjx_env.MENAGERIE_PATH = epath.Path(
        runfiles_repository("mujoco_menagerie_playground").as_posix()
    )


def _make_oracle(task_id: str) -> Any:
    _configure_oracle_assets()
    if task_id == "AlohaHandOver-v1":
        config = oracle_aloha_handover.default_config()
        config.impl = "jax"
        return oracle_aloha_handover.HandOver(config=config)
    if task_id == "AlohaSinglePegInsertion-v1":
        config = oracle_aloha_peg.default_config()
        config.impl = "jax"
        return oracle_aloha_peg.SinglePegInsertion(config=config)
    if task_id == "ApolloJoystickFlatTerrain-v1":
        config = oracle_apollo.default_config()
        config.impl = "jax"
        config.noise_config.level = 0.0
        config.push_config.enable = False
        config.command_config.min = [0.0, 0.0, 0.0]
        config.command_config.max = [0.0, 0.0, 0.0]
        config.command_config.zero_prob = [1.0, 1.0, 1.0]
        return oracle_apollo.Joystick(task="flat_terrain", config=config)
    if task_id == "BarkourJoystick-v1":
        config = oracle_barkour.default_config()
        config.impl = "jax"
        config.obs_noise = -1.0
        config.lin_vel_x = [0.0, 0.0]
        config.lin_vel_y = [0.0, 0.0]
        config.ang_vel_yaw = [0.0, 0.0]
        config.kick_wait_steps = [100000, 100001]
        return oracle_barkour.Joystick(config=config)
    if task_id == "Op3Joystick-v1":
        config = oracle_op3.default_config()
        config.impl = "jax"
        config.obs_noise = -1.0
        config.lin_vel_x = [0.0, 0.0]
        config.lin_vel_y = [0.0, 0.0]
        config.ang_vel_yaw = [0.0, 0.0]
        # The assets-dict loader strips directories and aliases OP3 visual
        # meshes with same-named collision meshes. Use the pinned file tree
        # for the entire oracle, including its independent physics rollout.
        xml = _find_runfile(
            "envpool/mujoco/playground/assets/mujoco_playground/_src/"
            "locomotion/op3/xmls/scene_mjx_feetonly.xml"
        )
        with mock.patch.object(
            mujoco.MjModel,
            "from_xml_string",
            return_value=mujoco.MjModel.from_xml_path(xml.as_posix()),
        ):
            return oracle_op3.Joystick(config=config)
    if task_id in (
        "BerkeleyHumanoidJoystickFlatTerrain-v1",
        "BerkeleyHumanoidJoystickRoughTerrain-v1",
    ):
        config = oracle_berkeley.default_config()
        config.impl = "jax"
        config.noise_config.level = 0.0
        config.push_config.enable = False
        config.lin_vel_x = [0.0, 0.0]
        config.lin_vel_y = [0.0, 0.0]
        config.ang_vel_yaw = [0.0, 0.0]
        task = "rough_terrain" if "RoughTerrain" in task_id else "flat_terrain"
        return oracle_berkeley.Joystick(task=task, config=config)
    if task_id in (
        "G1JoystickFlatTerrain-v1",
        "G1JoystickRoughTerrain-v1",
    ):
        config = oracle_g1.default_config()
        config.impl = "jax"
        config.noise_config.level = 0.0
        config.push_config.enable = False
        config.lin_vel_x = [0.0, 0.0]
        config.lin_vel_y = [0.0, 0.0]
        config.ang_vel_yaw = [0.0, 0.0]
        task = "rough_terrain" if "RoughTerrain" in task_id else "flat_terrain"
        return oracle_g1.Joystick(task=task, config=config)
    if task_id in (
        "T1JoystickFlatTerrain-v1",
        "T1JoystickRoughTerrain-v1",
    ):
        config = oracle_t1.default_config()
        config.impl = "jax"
        config.noise_config.level = 0.0
        config.push_config.enable = False
        config.lin_vel_x = [0.0, 0.0]
        config.lin_vel_y = [0.0, 0.0]
        config.ang_vel_yaw = [0.0, 0.0]
        task = "rough_terrain" if "RoughTerrain" in task_id else "flat_terrain"
        return oracle_t1.Joystick(task=task, config=config)
    if task_id == "H1InplaceGaitTracking-v1":
        config = oracle_h1_inplace.default_config()
        config.impl = "jax"
        config.obs_noise.level = 0.0
        return oracle_h1_inplace.InplaceGaitTracking(config=config)
    if task_id == "H1JoystickGaitTracking-v1":
        config = oracle_h1_joystick.default_config()
        config.impl = "jax"
        config.obs_noise.level = 0.0
        config.lin_vel_x = [0.0, 0.0]
        config.lin_vel_y = [0.0, 0.0]
        config.ang_vel_yaw = [0.0, 0.0]
        return oracle_h1_joystick.JoystickGaitTracking(config=config)
    if task_id == "SpotFlatTerrainJoystick-v1":
        config = oracle_spot_joystick.default_config()
        config.impl = "jax"
        config.obs_noise.scales.joint_pos = 0.0
        config.obs_noise.scales.gyro = 0.0
        config.obs_noise.scales.gravity = 0.0
        config.obs_noise.scales.feet_pos = [0.0, 0.0, 0.0]
        config.pert_config.enable = False
        config.command_config.lin_vel_x = [0.0, 0.0]
        config.command_config.lin_vel_y = [0.0, 0.0]
        config.command_config.ang_vel_yaw = [0.0, 0.0]
        return oracle_spot_joystick.Joystick(task="flat_terrain", config=config)
    if task_id == "SpotGetup-v1":
        config = oracle_spot_getup.default_config()
        config.impl = "jax"
        config.obs_noise.level = 0.0
        return oracle_spot_getup.Getup(config=config)
    if task_id == "SpotJoystickGaitTracking-v1":
        config = oracle_spot_gait.default_config()
        config.impl = "jax"
        config.obs_noise.scales.joint_pos = 0.0
        config.obs_noise.scales.gyro = 0.0
        config.obs_noise.scales.gravity = 0.0
        config.obs_noise.scales.feet_pos = [0.0, 0.0, 0.0]
        config.command_config.lin_vel_x = [0.0, 0.0]
        config.command_config.lin_vel_y = [0.0, 0.0]
        config.command_config.ang_vel_yaw = [0.0, 0.0]
        return oracle_spot_gait.JoystickGaitTracking(config=config)
    if task_id == "PandaPickCube-v1":
        config = oracle_panda_pick.default_config()
        config.impl = "jax"
        return oracle_panda_pick.PandaPickCube(config=config)
    if task_id == "PandaPickCubeCartesian-v1":
        config = oracle_panda_pick_cartesian.default_config()
        config.impl = "jax"
        return oracle_panda_pick_cartesian.PandaPickCubeCartesian(config=config)
    if task_id == "PandaPickCubeOrientation-v1":
        config = oracle_panda_pick.default_config()
        config.impl = "jax"
        return oracle_panda_pick.PandaPickCubeOrientation(config=config)
    if task_id == "PandaOpenCabinet-v1":
        config = oracle_panda_open_cabinet.default_config()
        config.impl = "jax"
        return oracle_panda_open_cabinet.PandaOpenCabinet(config=config)
    if task_id == "PandaRobotiqPushCube-v1":
        config = oracle_panda_robotiq.default_config()
        config.impl = "jax"
        config.noise_config.action_min_delay = 0
        config.noise_config.action_max_delay = 1
        config.noise_config.obs_min_delay = 0
        config.noise_config.obs_max_delay = 1
        config.noise_config.noise_scales.obj_pos = 0.0
        config.noise_config.noise_scales.obj_angle = 0.0
        config.noise_config.noise_scales.robot_qpos = 0.0
        config.noise_config.noise_scales.robot_qvel = 0.0
        config.noise_config.noise_scales.eef_pos = 0.0
        config.noise_config.noise_scales.eef_angle = 0.0
        return oracle_panda_robotiq.PandaRobotiqPushCube(config=config)
    if task_id == "LeapCubeRotateZAxis-v1":
        config = oracle_leap_rotate.default_config()
        config.impl = "jax"
        config.noise_config.level = 0.0
        return oracle_leap_rotate.CubeRotateZAxis(config=config)
    if task_id == "LeapCubeReorient-v1":
        config = oracle_leap_reorient.default_config()
        config.impl = "jax"
        config.obs_noise.level = 0.0
        config.pert_config.enable = False
        return oracle_leap_reorient.CubeReorient(config=config)
    if task_id == "AeroCubeRotateZAxis-v1":
        config = oracle_aero_rotate.default_config()
        config.noise_config.level = 0.0
        return oracle_aero_rotate.CubeRotateZAxis(config=config)
    if task_id == "Go1Getup-v1":
        config = oracle_go1_getup.default_config()
        config.impl = "jax"
        config.noise_config.level = 0.0
        return oracle_go1_getup.Getup(config=config)
    if task_id == "Go1Handstand-v1":
        config = oracle_go1_handstand.default_config()
        config.impl = "jax"
        config.noise_config.level = 0.0
        return oracle_go1_handstand.Handstand(config=config)
    if task_id == "Go1Footstand-v1":
        config = oracle_go1_handstand.default_config()
        config.impl = "jax"
        config.noise_config.level = 0.0
        return oracle_go1_handstand.Footstand(config=config)
    config = oracle_go1.default_config()
    config.impl = "jax"
    config.noise_config.level = 0.0
    config.command_config.a = [0.0, 0.0, 0.0]
    config.command_config.b = [0.0, 0.0, 0.0]
    task = "rough_terrain" if "RoughTerrain" in task_id else "flat_terrain"
    return oracle_go1.Joystick(task=task, config=config)


def _make_actions(steps: int, action_size: int) -> np.ndarray:
    rng = np.random.default_rng(20260513)
    return rng.uniform(-0.6, 0.6, size=(steps, 1, action_size)).astype(
        np.float64
    )


def _oracle_metric_name(info_key: str, task_id: str) -> str:
    key = info_key.removeprefix("reward_")
    if task_id == "PandaPickCubeCartesian-v1":
        return "reward/" + key
    if task_id.startswith(("Aloha", "Panda")):
        return key
    return "reward/" + key


def _env_config(task_id: str) -> dict[str, float]:
    if task_id.startswith("Aloha"):
        return {}
    if task_id.startswith("Apollo"):
        return _APOLLO_CONFIG
    if task_id.startswith("Barkour"):
        return _BARKOUR_CONFIG
    if task_id.startswith("Op3"):
        return _OP3_CONFIG
    if task_id.startswith("BerkeleyHumanoid"):
        return _BERKELEY_CONFIG
    if task_id.startswith("G1"):
        return _G1_CONFIG
    if task_id.startswith("T1"):
        return _T1_CONFIG
    if task_id.startswith("H1"):
        return _H1_CONFIG
    if task_id == "SpotFlatTerrainJoystick-v1":
        return _SPOT_JOYSTICK_CONFIG
    if task_id == "SpotGetup-v1":
        return _SPOT_GETUP_CONFIG
    if task_id == "SpotJoystickGaitTracking-v1":
        return _SPOT_GAIT_CONFIG
    if task_id == "PandaPickCubeCartesian-v1":
        return {"guide_sample_prob": 0.0}
    if task_id == "PandaRobotiqPushCube-v1":
        return {
            "action_min_delay": 0,
            "action_max_delay": 1,
            "obs_min_delay": 0,
            "obs_max_delay": 1,
            "noise_obj_pos": 0.0,
            "noise_obj_angle": 0.0,
            "noise_robot_qpos": 0.0,
            "noise_robot_qvel": 0.0,
            "noise_eef_pos": 0.0,
            "noise_eef_angle": 0.0,
        }
    if task_id in (
        "LeapCubeRotateZAxis-v1",
        "LeapCubeReorient-v1",
        "AeroCubeRotateZAxis-v1",
    ):
        return {"noise_level": 0.0, "pert_enable": 0.0}
    if task_id.startswith("Panda"):
        return {}
    if "Joystick" in task_id:
        return _JOYSTICK_CONFIG
    return _CONFIG


def _reward_info_keys(task_id: str) -> tuple[str, ...]:
    if task_id == "AlohaHandOver-v1":
        return _ALOHA_HANDOVER_REWARD_INFO_KEYS
    if task_id == "AlohaSinglePegInsertion-v1":
        return _ALOHA_PEG_REWARD_INFO_KEYS
    if task_id.startswith("Apollo"):
        return _APOLLO_REWARD_INFO_KEYS
    if task_id.startswith("Barkour"):
        return _BARKOUR_REWARD_INFO_KEYS
    if task_id.startswith("Op3"):
        return _OP3_REWARD_INFO_KEYS
    if task_id.startswith("BerkeleyHumanoid"):
        return _BERKELEY_REWARD_INFO_KEYS
    if task_id.startswith("G1"):
        return _G1_REWARD_INFO_KEYS
    if task_id.startswith("T1"):
        return _T1_REWARD_INFO_KEYS
    if task_id == "H1InplaceGaitTracking-v1":
        return _H1_INPLACE_REWARD_INFO_KEYS
    if task_id == "H1JoystickGaitTracking-v1":
        return _H1_JOYSTICK_REWARD_INFO_KEYS
    if task_id == "SpotFlatTerrainJoystick-v1":
        return _SPOT_JOYSTICK_REWARD_INFO_KEYS
    if task_id == "SpotGetup-v1":
        return _SPOT_GETUP_REWARD_INFO_KEYS
    if task_id == "SpotJoystickGaitTracking-v1":
        return _SPOT_GAIT_REWARD_INFO_KEYS
    if task_id in ("PandaPickCube-v1", "PandaPickCubeOrientation-v1"):
        return _PANDA_PICK_REWARD_INFO_KEYS
    if task_id == "PandaPickCubeCartesian-v1":
        return _PANDA_CARTESIAN_REWARD_INFO_KEYS
    if task_id == "PandaOpenCabinet-v1":
        return _PANDA_OPEN_CABINET_REWARD_INFO_KEYS
    if task_id == "PandaRobotiqPushCube-v1":
        return _PANDA_ROBOTIQ_REWARD_INFO_KEYS
    if task_id in ("LeapCubeRotateZAxis-v1", "AeroCubeRotateZAxis-v1"):
        return _HAND_ROTATE_REWARD_INFO_KEYS
    if task_id == "LeapCubeReorient-v1":
        return _LEAP_REORIENT_REWARD_INFO_KEYS
    if "Joystick" in task_id:
        return _REWARD_INFO_KEYS
    if "Getup" in task_id:
        return _GETUP_REWARD_INFO_KEYS
    return _HANDSTAND_REWARD_INFO_KEYS


def _sync_oracle_state(
    oracle: Any,
    state: oracle_mjx_env.State,
    info: dict[str, Any],
) -> oracle_mjx_env.State:
    if isinstance(oracle, oracle_go1.Joystick):
        qpos = jnp.asarray(info["qpos"][0, : oracle.mj_model.nq])
        qvel = jnp.asarray(info["qvel"][0, : oracle.mj_model.nv])
        ctrl = jnp.asarray(info["ctrl"][0, : oracle.mj_model.nu])
        data = state.data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        data = mjx.forward(oracle.mjx_model, data)
        if "sensordata" in info:
            # Python MuJoCo matches EnvPool's reset sensors bitwise, but MJX
            # JAX `forward` has a small reset-time accelerometer residual while
            # feet are in active floor contact. Sensor data is a derived MuJoCo
            # cache, so include it in reset-time oracle sync instead of
            # loosening obs tolerances.
            data = data.replace(
                sensordata=jnp.asarray(
                    info["sensordata"][0, : data.sensordata.size]
                )
            )
    else:
        mujoco_data = _make_mujoco_data_from_info(oracle, info)
        data = _sync_mjx_data_from_mujoco(
            state.data, mujoco_data, oracle.mj_model
        )
    oracle_info = dict(state.info)
    if "command" in info:
        oracle_info["command"] = jnp.asarray(info["command"][0])
    if "steps_until_next_cmd" in info:
        oracle_info["steps_until_next_cmd"] = jnp.asarray(
            info["steps_until_next_cmd"][0], dtype=jnp.int32
        )
    for key in (
        "last_act",
        "last_last_act",
        "motor_targets",
        "last_vel",
        "feet_air_time",
        "last_contact",
        "swing_peak",
        "qpos_error_history",
        "qvel_history",
        "lin_vel",
        "ang_vel",
        "phase",
        "phase_dt",
        "gait_freq",
        "foot_height",
        "push",
        "kick_dir",
        "vel_kick",
        "last_kick_step",
        "filtered_linvel",
        "filtered_angvel",
        "target_pos",
        "reached_box",
        "previously_gripped",
        "prev_reward",
        "current_pos",
        "last_action",
        "action_history",
        "obs_history",
        "cube_pos_error_history",
        "cube_ori_error_history",
        "goal_quat_dquat",
        "pert_dir",
        "prev_action",
        "no_soln",
        "prev_potential",
        "episode_picked",
    ):
        if key in info:
            oracle_info[key] = jnp.asarray(info[key][0])
    if "steps" in info:
        oracle_info["_steps"] = jnp.asarray(info["steps"][0], dtype=jnp.int32)
    for key in (
        "step",
        "push_step",
        "push_interval_steps",
        "kick_wait_steps",
        "kick_duration_steps",
        "gait",
        "success_step_count",
        "steps_since_last_success",
        "success_count",
        "prev_step_success",
        "curriculum_id",
    ):
        if key in info:
            oracle_info[key] = jnp.asarray(info[key][0], dtype=jnp.int32)
    if isinstance(
        oracle,
        (
            oracle_h1_inplace.InplaceGaitTracking,
            oracle_h1_joystick.JoystickGaitTracking,
        ),
    ):
        history_size = int(oracle._config.history_len) * 19  # pylint: disable=protected-access
        for key in ("qpos_error_history", "qvel_history"):
            if key in info:
                history = np.zeros(history_size, dtype=np.float64)
                history[: history_size - 19] = info[key][0, 19:history_size]
                oracle_info[key] = jnp.asarray(history)
    if isinstance(
        oracle,
        (
            oracle_spot_joystick.Joystick,
            oracle_spot_gait.JoystickGaitTracking,
        ),
    ):
        history_size = int(oracle._config.history_len) * 12  # pylint: disable=protected-access
        history = np.zeros(history_size, dtype=np.float64)
        history[: history_size - 12] = info["qpos_error_history"][
            0, 12:history_size
        ]
        oracle_info["qpos_error_history"] = jnp.asarray(history)
    if isinstance(oracle, oracle_panda_robotiq.PandaRobotiqPushCube):
        action_history_size = int(oracle._config.action_history_len) * 7  # pylint: disable=protected-access
        obs_history_size = (
            int(oracle._config.obs_history_len) * 48  # pylint: disable=protected-access
        )
        oracle_info["action_history"] = jnp.asarray(
            info["action_history"][0, :action_history_size]
        )
        oracle_info["obs_history"] = jnp.asarray(
            info["obs_history"][0, :obs_history_size]
        )
    if isinstance(
        oracle,
        (
            oracle_leap_rotate.CubeRotateZAxis,
            oracle_leap_reorient.CubeReorient,
            oracle_aero_rotate.CubeRotateZAxis,
        ),
    ):
        for key in ("last_act", "last_last_act", "motor_targets"):
            oracle_info[key] = jnp.asarray(info[key][0, : oracle.action_size])
    if isinstance(oracle, (oracle_barkour.Joystick, oracle_op3.Joystick)):
        obs = jnp.asarray(info["obs_history"][0])
    elif isinstance(oracle, oracle_berkeley.Joystick):
        contact = _berkeley_contact(oracle, data)
        obs = oracle._get_obs(  # pylint: disable=protected-access
            data, oracle_info, contact
        )
    elif isinstance(oracle, oracle_g1.Joystick):
        contact = _g1_contact(oracle, data)
        obs = oracle._get_obs(  # pylint: disable=protected-access
            data, oracle_info, contact
        )
    elif isinstance(oracle, oracle_t1.Joystick):
        contact = _t1_contact(oracle, data)
        obs = oracle._get_obs(  # pylint: disable=protected-access
            data, oracle_info, contact
        )
    elif isinstance(
        oracle,
        (
            oracle_h1_inplace.InplaceGaitTracking,
            oracle_h1_joystick.JoystickGaitTracking,
        ),
    ):
        contact = _h1_contact_by_adr(oracle, data)
        obs = oracle._get_obs(  # pylint: disable=protected-access
            data, oracle_info, jax.random.PRNGKey(0), contact
        )
    elif isinstance(oracle, oracle_spot_joystick.Joystick):
        obs = oracle._get_obs(  # pylint: disable=protected-access
            data, oracle_info, jax.random.PRNGKey(0)
        )
    elif isinstance(oracle, oracle_spot_getup.Getup):
        obs = oracle._get_obs(  # pylint: disable=protected-access
            data, oracle_info, jax.random.PRNGKey(0)
        )
    elif isinstance(oracle, oracle_spot_gait.JoystickGaitTracking):
        contact = _spot_contact(oracle, data)
        obs = oracle._get_obs(  # pylint: disable=protected-access
            data, oracle_info, jax.random.PRNGKey(0), contact
        )
    elif isinstance(oracle, oracle_go1_handstand.Handstand):
        contact = _handstand_unwanted_contact(oracle, data)
        obs = oracle._get_obs(  # pylint: disable=protected-access
            data, oracle_info, contact
        )
    elif isinstance(oracle, oracle_aloha_peg.SinglePegInsertion):
        obs = oracle._get_obs(data)  # pylint: disable=protected-access
    elif isinstance(oracle, oracle_panda_pick_cartesian.PandaPickCubeCartesian):
        obs = oracle._get_obs(  # pylint: disable=protected-access
            data, oracle_info
        )
        obs = jnp.concat(
            [
                obs,
                jnp.reshape(jnp.asarray(oracle_info["no_soln"]), (1,)),
                jnp.asarray(oracle_info["prev_action"]),
            ],
            axis=0,
        )
    elif isinstance(oracle, oracle_panda_robotiq.PandaRobotiqPushCube):
        obs = oracle._get_single_obs(  # pylint: disable=protected-access
            data, oracle_info
        )
    elif isinstance(
        oracle,
        (
            oracle_leap_rotate.CubeRotateZAxis,
            oracle_aero_rotate.CubeRotateZAxis,
        ),
    ):
        obs_history_size = int(oracle._config.history_len) * (  # pylint: disable=protected-access
            14 if isinstance(oracle, oracle_aero_rotate.CubeRotateZAxis) else 32
        )
        obs = oracle._get_obs(  # pylint: disable=protected-access
            data,
            oracle_info,
            jnp.asarray(info["obs_history"][0, :obs_history_size]),
        )
    elif isinstance(oracle, oracle_leap_reorient.CubeReorient):
        history_len = int(oracle._config.history_len)  # pylint: disable=protected-access
        oracle_info["qpos_error_history"] = jnp.asarray(
            info["qpos_error_history"][0, : history_len * 16]
        )
        oracle_info["cube_pos_error_history"] = jnp.asarray(
            info["cube_pos_error_history"][0, : history_len * 3]
        )
        oracle_info["cube_ori_error_history"] = jnp.asarray(
            info["cube_ori_error_history"][0, : history_len * 6]
        )
        obs = oracle._get_obs(  # pylint: disable=protected-access
            data, oracle_info
        )
    else:
        obs = oracle._get_obs(  # pylint: disable=protected-access
            data, oracle_info
        )
    return state.replace(data=data, obs=obs, info=oracle_info)


def _make_mujoco_data_from_info(
    oracle: Any, info: dict[str, Any]
) -> mujoco.MjData:
    return _make_mujoco_data_from_model(
        oracle.mj_model,
        info,
        copy_derived=not isinstance(oracle, oracle_go1.Joystick),
    )


def _make_mujoco_data_from_model(
    model: mujoco.MjModel, info: dict[str, Any], *, copy_derived: bool = True
) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.qpos[:] = info["qpos"][0, : model.nq]
    data.qvel[:] = info["qvel"][0, : model.nv]
    data.ctrl[:] = info["ctrl"][0, : model.nu]
    if "mocap_pos" in info:
        data.mocap_pos[:] = info["mocap_pos"][0, : model.nmocap * 3].reshape(
            model.nmocap, 3
        )
    if "mocap_quat" in info:
        data.mocap_quat[:] = info["mocap_quat"][0, : model.nmocap * 4].reshape(
            model.nmocap, 4
        )
    if "xfrc_applied" in info:
        data.xfrc_applied[:] = info["xfrc_applied"][
            0, : model.nbody * 6
        ].reshape(model.nbody, 6)
    if copy_derived and "qacc_warmstart" in info:
        data.qacc_warmstart[:] = info["qacc_warmstart"][0, : model.nv]
    mujoco.mj_forward(model, data)
    if not copy_derived:
        return data
    if "qacc" in info:
        data.qacc[:] = info["qacc"][0, : model.nv]
    if "qacc_warmstart" in info:
        data.qacc_warmstart[:] = info["qacc_warmstart"][0, : model.nv]
    if "sensordata" in info:
        data.sensordata[:] = info["sensordata"][0, : data.sensordata.size]
    return data


def _sync_mjx_data_from_mujoco(
    state_data: mjx.Data, data: mujoco.MjData, model: mujoco.MjModel
) -> mjx.Data:
    return state_data.replace(
        time=jnp.asarray(data.time),
        act=jnp.asarray(data.act),
        qfrc_applied=jnp.asarray(data.qfrc_applied),
        qpos=jnp.asarray(data.qpos),
        qvel=jnp.asarray(data.qvel),
        qacc=jnp.asarray(data.qacc),
        qacc_warmstart=jnp.asarray(data.qacc_warmstart),
        ctrl=jnp.asarray(data.ctrl),
        sensordata=jnp.asarray(data.sensordata),
        xpos=jnp.asarray(np.asarray(data.xpos).reshape(model.nbody, 3)),
        xquat=jnp.asarray(np.asarray(data.xquat).reshape(model.nbody, 4)),
        xmat=jnp.asarray(np.asarray(data.xmat).reshape(model.nbody, 3, 3)),
        geom_xpos=jnp.asarray(
            np.asarray(data.geom_xpos).reshape(model.ngeom, 3)
        ),
        geom_xmat=jnp.asarray(
            np.asarray(data.geom_xmat).reshape(model.ngeom, 3, 3)
        ),
        site_xpos=jnp.asarray(
            np.asarray(data.site_xpos).reshape(model.nsite, 3)
        ),
        site_xmat=jnp.asarray(
            np.asarray(data.site_xmat).reshape(model.nsite, 3, 3)
        ),
        actuator_force=jnp.asarray(data.actuator_force),
        qfrc_actuator=jnp.asarray(data.qfrc_actuator),
        xfrc_applied=jnp.asarray(
            np.asarray(data.xfrc_applied).reshape(model.nbody, 6)
        ),
        mocap_pos=jnp.asarray(
            np.asarray(data.mocap_pos).reshape(model.nmocap, 3)
        ),
        mocap_quat=jnp.asarray(
            np.asarray(data.mocap_quat).reshape(model.nmocap, 4)
        ),
    )


def _handstand_unwanted_contact(
    oracle: oracle_go1_handstand.Handstand, data: mjx.Data
) -> jax.Array:
    return jnp.array([
        data.sensordata[oracle._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in oracle._fullcollision_floor_found_sensor
    ])


def _berkeley_contact(
    oracle: oracle_berkeley.Joystick, data: mjx.Data
) -> jax.Array:
    return jnp.array([
        data.sensordata[oracle._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in oracle._feet_floor_found_sensor
    ])


def _g1_contact(oracle: oracle_g1.Joystick, data: mjx.Data) -> jax.Array:
    return jnp.array([
        data.sensordata[oracle._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in oracle._feet_floor_found_sensor
    ])


def _t1_contact(oracle: oracle_t1.Joystick, data: mjx.Data) -> jax.Array:
    left = jnp.array([
        data.sensordata[oracle._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in oracle._left_foot_floor_found_sensor
    ])
    right = jnp.array([
        data.sensordata[oracle._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in oracle._right_foot_floor_found_sensor
    ])
    return jnp.hstack([jnp.any(left), jnp.any(right)])


def _spot_contact(oracle: Any, data: mjx.Data) -> jax.Array:
    return jnp.array([
        data.sensordata[oracle._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in oracle._feet_floor_found_sensor
    ])


def _h1_contact_by_adr(oracle: Any, data: mjx.Data) -> jax.Array:
    left = jnp.array([
        data.sensordata[oracle._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in oracle._left_foot_floor_found_sensor
    ])
    right = jnp.array([
        data.sensordata[oracle._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in oracle._right_foot_floor_found_sensor
    ])
    return jnp.hstack([jnp.any(left), jnp.any(right)])


def _step_mujoco_oracle(
    oracle: Any,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    """Run the official task step with an independent MuJoCo C integrator."""

    def step(
        model: Any, previous: mjx.Data, ctrl: Any, n_substeps: int = 1
    ) -> mjx.Data:
        del model
        # Carry only oracle-owned changes (pushes, targets, activation state)
        # into physics. No EnvPool state is read after the initial reset sync.
        for field in (
            "qpos",
            "qvel",
            "act",
            "qacc_warmstart",
            "mocap_pos",
            "mocap_quat",
            "xfrc_applied",
            "qfrc_applied",
        ):
            getattr(data, field)[:] = np.asarray(getattr(previous, field))
        data.time = float(previous.time)
        data.ctrl[:] = np.asarray(ctrl)
        for _ in range(n_substeps):
            mujoco.mj_step(oracle.mj_model, data)
        return _sync_mjx_data_from_mujoco(previous, data, oracle.mj_model)

    with mock.patch.object(oracle_mjx_env, "step", step):
        return oracle.step(state, jnp.asarray(action))


@contextmanager
def _oracle_physics(task_id: str) -> Iterator[Any]:
    """Keep sensitive mesh contacts on one C engine, with checked models."""
    oracle = _make_oracle(task_id)
    if platform.system() != "Linux" or not isinstance(
        oracle, (oracle_g1.Joystick, oracle_berkeley.Joystick)
    ):
        yield oracle
        return

    # Linux wheels use libc++, while the native build uses libstdc++.
    # MjSpec shares C++ strings, so swapping its library is unsafe. Use the C
    # API for physics and keep the Python bindings with their matching library.
    import envpool.mujoco.playground_envpool as playground_envpool

    library = ctypes.CDLL(playground_envpool.__file__)
    library.mj_loadXML.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    library.mj_loadXML.restype = ctypes.c_void_p
    for name in ("mj_step", "mj_forward"):
        getattr(library, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        getattr(library, name).restype = None
    relative = Path(oracle.xml_path).relative_to(oracle_mjx_env.ROOT_PATH)
    xml = _find_runfile(
        "envpool/mujoco/playground/assets/mujoco_playground/_src/"
        + relative.as_posix()
    )
    error = ctypes.create_string_buffer(4096)
    pointer = library.mj_loadXML(os.fsencode(xml), None, error, len(error))
    if not pointer:
        raise ValueError(error.value.decode())
    model = mujoco.MjModel._from_model_ptr(pointer)
    model.opt.timestep = oracle.mj_model.opt.timestep
    # Independently compiled models must describe the same task. Float32 mesh
    # normals differ by <3e-8; double derived fields differ by <3e-14. These tiny
    # residuals can be amplified during mesh contact dynamics.
    for name in dir(model):
        value = getattr(model, name)
        if isinstance(value, np.ndarray):
            expected = getattr(oracle.mj_model, name)
            if np.issubdtype(value.dtype, np.floating):
                tolerance = 1e-7 if value.dtype.itemsize == 4 else 1e-13
                np.testing.assert_allclose(
                    value,
                    expected,
                    rtol=tolerance,
                    atol=tolerance,
                    err_msg=name,
                )
            else:
                np.testing.assert_array_equal(value, expected, err_msg=name)
    for name in dir(model.opt):
        if not name.startswith("_"):
            np.testing.assert_equal(
                getattr(model.opt, name),
                getattr(oracle.mj_model.opt, name),
                err_msg=name,
            )
    model.vis.global_.offwidth = oracle.mj_model.vis.global_.offwidth
    model.vis.global_.offheight = oracle.mj_model.vis.global_.offheight
    oracle._mj_model = model
    oracle._mjx_model = mjx.put_model(model, impl=oracle._config.impl)
    with (
        mock.patch.object(
            mujoco,
            "mj_step",
            lambda m, d: library.mj_step(m._address, d._address),
        ),
        mock.patch.object(
            mujoco,
            "mj_forward",
            lambda m, d: library.mj_forward(m._address, d._address),
        ),
    ):
        yield oracle


def _make_rough_oracle_for_mjx_diagnostic(
    *,
    iterations: int = 1,
    disable_frictionloss: bool = False,
) -> oracle_go1.Joystick:
    oracle = _make_oracle("Go1JoystickRoughTerrain-v1")
    oracle.mj_model.opt.iterations = iterations
    if disable_frictionloss:
        oracle.mj_model.opt.disableflags |= int(
            mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
        )
    # The upstream env builds the MJX model during construction. Rebuild it
    # after the diagnostic mutates MuJoCo options so both backends see exactly
    # the same model.
    oracle._mjx_model = mjx.put_model(  # pylint: disable=protected-access
        oracle.mj_model,
        impl=oracle.mjx_model.impl.value,
    )
    return oracle


def _compare_mujoco_c_and_mjx_one_control_step(
    oracle: oracle_go1.Joystick,
    *,
    raise_z: float = 1.0,
) -> dict[str, Any]:
    state = oracle.reset(jax.random.PRNGKey(0))
    qpos = np.asarray(state.data.qpos).copy()
    qvel = np.asarray(state.data.qvel).copy()
    ctrl = np.asarray(state.data.ctrl).copy()
    qpos[2] += raise_z

    mujoco_data = mujoco.MjData(oracle.mj_model)
    mujoco_data.qpos[:] = qpos
    mujoco_data.qvel[:] = qvel
    mujoco_data.ctrl[:] = ctrl
    mujoco.mj_forward(oracle.mj_model, mujoco_data)

    mjx_data = state.data.replace(
        qpos=jnp.asarray(qpos),
        qvel=jnp.asarray(qvel),
        ctrl=jnp.asarray(ctrl),
    )
    mjx_data = mjx.forward(oracle.mjx_model, mjx_data)
    initial_mujoco_ncon = int(mujoco_data.ncon)
    initial_mjx_ncon = int(np.asarray(mjx_data._impl.ncon))

    action = np.linspace(-0.6, 0.6, oracle.action_size, dtype=np.float64)
    motor_targets = (
        np.asarray(oracle._default_pose)  # pylint: disable=protected-access
        + action * oracle._config.action_scale  # pylint: disable=protected-access
    )
    mujoco_data.ctrl[:] = motor_targets
    mjx_data = mjx_data.replace(ctrl=jnp.asarray(motor_targets))
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, mujoco_data)
        mjx_data = mjx.step(oracle.mjx_model, mjx_data)

    return {
        "initial_mujoco_ncon": initial_mujoco_ncon,
        "initial_mjx_ncon": initial_mjx_ncon,
        "final_mujoco_ncon": int(mujoco_data.ncon),
        "final_mjx_ncon": int(np.asarray(mjx_data._impl.ncon)),
        "qpos": float(
            np.max(np.abs(np.asarray(mjx_data.qpos) - mujoco_data.qpos))
        ),
        "qvel": float(
            np.max(np.abs(np.asarray(mjx_data.qvel) - mujoco_data.qvel))
        ),
        "qacc": float(
            np.max(np.abs(np.asarray(mjx_data.qacc) - mujoco_data.qacc))
        ),
        "sensordata": float(
            np.max(
                np.abs(np.asarray(mjx_data.sensordata) - mujoco_data.sensordata)
            )
        ),
    }


def _rollout_atol(task_id: str) -> float:
    # Float32 IK targets differ by <= 6e-8; after 128 steps their largest
    # velocity residual is 9.2e-7 and reward residual is 1.9e-7.
    if task_id in ("PandaPickCubeCartesian-v1", "PandaRobotiqPushCube-v1"):
        return 1e-6
    # Hand contacts amplify control roundoff to < 3e-9 over a full episode;
    # the float32 reward can differ by one rounding unit (7.5e-9).
    if task_id == "AeroCubeRotateZAxis-v1":
        return 1e-8
    return 5e-11


def _assert_obs_close(
    test: absltest.TestCase,
    task_id: str,
    actual: Any,
    expected: Any,
) -> None:
    expected_state = (
        expected["state"] if isinstance(expected, dict) else expected
    )
    if isinstance(actual, dict):
        expected_keys = {"state"}
        if isinstance(expected, dict) and "privileged_state" in expected:
            expected_keys.add("privileged_state")
        test.assertEqual(set(actual.keys()), expected_keys)
        actual_state = actual["state"][0]
    else:
        test.assertNotIsInstance(expected, dict)
        actual_state = actual[0]
    np.testing.assert_allclose(
        np.asarray(actual_state),
        np.asarray(expected_state),
        atol=_rollout_atol(task_id),
        rtol=_rollout_atol(task_id),
        err_msg="obs[state]",
    )

    if not isinstance(actual, dict) or "privileged_state" not in actual:
        return
    actual_privileged = np.asarray(actual["privileged_state"][0])
    expected_privileged = np.asarray(expected["privileged_state"])
    np.testing.assert_allclose(
        actual_privileged,
        expected_privileged,
        atol=_rollout_atol(task_id),
        rtol=_rollout_atol(task_id),
        err_msg="obs[privileged_state]",
    )


def _assert_mujoco_state_equal(
    task_id: str,
    info: dict[str, Any],
    oracle_state: oracle_mjx_env.State,
    oracle: Any,
) -> None:
    for key, size in (
        ("qpos", oracle.mj_model.nq),
        ("qvel", oracle.mj_model.nv),
        ("ctrl", oracle.mj_model.nu),
    ):
        atol = _rollout_atol(task_id)
        np.testing.assert_allclose(
            info[key][0, :size],
            np.asarray(getattr(oracle_state.data, key)),
            atol=atol,
            rtol=atol,
            err_msg=key,
        )


def _assert_frames_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    max_mean_abs_diff: float = 1.0,
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"frame shapes differ: {actual.shape} != {expected.shape}"
        )
    # Bound frame-wide visual error. Counting individual edge pixels made the
    # test depend on rasterization details despite matching independent state.
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    if diff.size and diff.mean() > max_mean_abs_diff:
        raise AssertionError(
            f"mean render delta {diff.mean():.3f} exceeded {max_mean_abs_diff:.3f}"
        )


def _official_render_camera(task_id: str) -> str | int:
    if task_id.startswith((
        "Apollo",
        "Barkour",
        "Berkeley",
        "G1",
        "Go1",
        "H1",
        "Op3",
        "Spot",
        "T1",
    )):
        return "track"
    return -1


class _OfficialFrameRenderer:
    def __init__(self, oracle: Any, task_id: str):
        model = oracle.mj_model
        # Match native buffer dimensions: skybox projection on CGL depends on
        # the full offscreen buffer, not just the requested output viewport.
        model.vis.global_.offwidth = _RENDER_WIDTH
        model.vis.global_.offheight = _RENDER_HEIGHT
        self._camera = _official_render_camera(task_id)
        self._renderer = mujoco.Renderer(
            model, height=_RENDER_HEIGHT, width=_RENDER_WIDTH
        )

    def render(self, data: mujoco.MjData) -> np.ndarray:
        # MuJoCo builds infinite-plane geometry before updating the camera.
        # Refresh that geometry at the selected camera before the single draw.
        self._renderer.update_scene(data, camera=self._camera)
        self._renderer.update_scene(data, camera=self._camera)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()


def _assert_rollout_outputs_close(
    test: absltest.TestCase,
    task_id: str,
    obs: Any,
    reward: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    info: dict[str, Any],
    state: oracle_mjx_env.State,
    oracle: Any,
) -> None:
    _assert_mujoco_state_equal(task_id, info, state, oracle)
    _assert_obs_close(test, task_id, obs, state.obs)
    # Compare the public float32 reward after the same dtype conversion.
    reward_report = {
        key: (
            float(np.asarray(info[key][0])),
            float(np.asarray(state.metrics[_oracle_metric_name(key, task_id)])),
        )
        for key in _reward_info_keys(task_id)
    }
    np.testing.assert_allclose(
        reward[0],
        np.asarray(state.reward, dtype=reward.dtype),
        atol=_rollout_atol(task_id),
        rtol=_rollout_atol(task_id),
        err_msg=f"reward components={reward_report}",
    )
    test.assertEqual(bool(terminated[0]), bool(state.done))
    test.assertFalse(bool(truncated[0]))
    if "command" in info:
        np.testing.assert_array_equal(
            info["command"][0],
            np.asarray(state.info["command"]),
        )
    # A command deadline sampled after termination cannot affect this episode.
    # Its RNG draw need not match across implementations; live deadlines do.
    if "steps_until_next_cmd" in info and not bool(state.done):
        test.assertEqual(
            int(info["steps_until_next_cmd"][0]),
            int(np.asarray(state.info["steps_until_next_cmd"])),
        )
    test.assertEqual(bool(info["terminated"][0]), bool(np.asarray(state.done)))
    for key in _reward_info_keys(task_id):
        np.testing.assert_allclose(
            info[key][0],
            np.asarray(state.metrics[_oracle_metric_name(key, task_id)]),
            atol=_rollout_atol(task_id),
            rtol=_rollout_atol(task_id),
            err_msg=f"info[{key}]",
        )


class PlaygroundOracleAlignTest(absltest.TestCase):
    """Compare the native EnvPool port against the pinned Playground oracle."""

    def test_registry_matches_upstream_non_dmc_tasks(self) -> None:
        """Pins coverage to Playground locomotion/manipulation, excluding DMC."""
        upstream_non_dmc = set(oracle_locomotion.ALL_ENVS) | set(
            oracle_manipulation.ALL_ENVS
        )
        envpool_tasks = set(playground_registration.PLAYGROUND_ENVS)
        self.assertLen(playground_registration.PLAYGROUND_ENVS, 29)
        self.assertLen(
            playground_registration.PLAYGROUND_ENVS, len(envpool_tasks)
        )
        self.assertEqual(envpool_tasks, upstream_non_dmc)
        self.assertEmpty(
            envpool_tasks
            & {
                "AcrobotSwingup",
                "CartpoleBalance",
                "CheetahRun",
                "FingerSpin",
                "HumanoidRun",
                "WalkerWalk",
            }
        )

    def test_mjx_frictionloss_gap_is_not_envpool_drift(self) -> None:
        """Documents why MuJoCo C, not MJX, is the step-level oracle here."""
        if not _is_primary_shard():
            return
        official = _compare_mujoco_c_and_mjx_one_control_step(
            _make_rough_oracle_for_mjx_diagnostic()
        )
        self.assertEqual(official["initial_mujoco_ncon"], 0)
        self.assertEqual(official["final_mujoco_ncon"], 0)
        self.assertGreater(official["initial_mjx_ncon"], 0)
        self.assertGreater(official["sensordata"], 1e-4)
        self.assertGreater(official["qacc"], 1e-2)

        no_frictionloss = _compare_mujoco_c_and_mjx_one_control_step(
            _make_rough_oracle_for_mjx_diagnostic(disable_frictionloss=True)
        )
        high_iterations = _compare_mujoco_c_and_mjx_one_control_step(
            _make_rough_oracle_for_mjx_diagnostic(iterations=5)
        )
        for label, report in (
            ("frictionloss disabled", no_frictionloss),
            ("iterations=5", high_iterations),
        ):
            with self.subTest(label=label):
                self.assertEqual(report["initial_mujoco_ncon"], 0)
                self.assertEqual(report["final_mujoco_ncon"], 0)
                for key in ("qpos", "qvel", "qacc", "sensordata"):
                    self.assertLessEqual(
                        report[key],
                        _STRICT_MJX_DIAGNOSTIC_ATOL,
                        msg=f"{label} {key}: {report}",
                    )

    def test_rollout_and_render_align_with_oracle(self) -> None:
        """Checks independent dynamics and rendering over nontrivial rollouts."""
        for task_id in _TASK_IDS:
            with (
                self.subTest(task_id=task_id),
                _oracle_physics(task_id) as oracle,
            ):
                env = make_gymnasium(
                    task_id,
                    num_envs=1,
                    seed=0,
                    render_mode="rgb_array",
                    render_width=_RENDER_WIDTH,
                    render_height=_RENDER_HEIGHT,
                    **_env_config(task_id),
                )
                actions = _make_actions(_ROLLOUT_STEPS, oracle.action_size)
                official_renderer: _OfficialFrameRenderer | None = None
                try:
                    obs, info = env.reset()
                    state = oracle.reset(jax.random.PRNGKey(0))
                    state = _sync_oracle_state(oracle, state, info)
                    mujoco_data = _make_mujoco_data_from_info(oracle, info)
                    _assert_obs_close(self, task_id, obs, state.obs)
                    official_renderer = _OfficialFrameRenderer(oracle, task_id)
                    rendered = env.render(env_ids=[0])
                    assert rendered is not None
                    _assert_frames_close(
                        rendered[0],
                        official_renderer.render(mujoco_data),
                    )

                    for step, action in enumerate(actions, start=1):
                        state = _step_mujoco_oracle(
                            oracle, state, action[0], mujoco_data
                        )
                        obs, reward, terminated, truncated, info = env.step(
                            action
                        )
                        with self.subTest(step=step):
                            _assert_rollout_outputs_close(
                                self,
                                task_id,
                                obs,
                                reward,
                                terminated,
                                truncated,
                                info,
                                state,
                                oracle,
                            )
                            rendered = env.render(env_ids=[0])
                            assert rendered is not None
                            _assert_frames_close(
                                rendered[0],
                                official_renderer.render(mujoco_data),
                            )
                        if bool(state.done):
                            break
                finally:
                    if official_renderer is not None:
                        official_renderer.close()
                    env.close()


if __name__ == "__main__":
    # Tasks are sharded above; Abseil would shard the methods a second time.
    unittest.main()
