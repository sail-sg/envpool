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
from functools import lru_cache
from pathlib import Path
from typing import Any

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

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

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from absl.testing import absltest
from etils import epath
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
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
from envpool.registration import make_gymnasium


def _configure_macos_mujoco_renderer() -> None:
    if platform.system() != "Darwin":
        return

    os.environ["MUJOCO_GL"] = "cgl"
    from mujoco import gl_context as mujoco_gl_context

    class _CglContext:
        def __init__(self, width: int, height: int):
            del width, height
            from mujoco.cgl import cgl

            self._pixel_format: Any = None
            self._context: Any = None
            self._locked = False
            attrib = cgl.CGLPixelFormatAttribute
            profile = cgl.CGLOpenGLProfile
            preferred_attribs = (
                attrib.CGLPFAOpenGLProfile,
                profile.CGLOGLPVersion_Legacy,
                attrib.CGLPFAColorSize,
                24,
                attrib.CGLPFAAlphaSize,
                8,
                attrib.CGLPFADepthSize,
                24,
                attrib.CGLPFAStencilSize,
                8,
                attrib.CGLPFAMultisample,
                attrib.CGLPFASampleBuffers,
                1,
                attrib.CGLPFASample,
                4,
                attrib.CGLPFAAccelerated,
                0,  # terminator
            )
            offline_attribs = (
                attrib.CGLPFAOpenGLProfile,
                profile.CGLOGLPVersion_Legacy,
                attrib.CGLPFAColorSize,
                24,
                attrib.CGLPFAAlphaSize,
                8,
                attrib.CGLPFADepthSize,
                24,
                attrib.CGLPFAStencilSize,
                8,
                attrib.CGLPFAAllowOfflineRenderers,
                0,  # value
                0,  # terminator
            )
            if not self._choose_pixel_format(
                cgl, preferred_attribs
            ) and not self._choose_pixel_format(cgl, offline_attribs):
                raise RuntimeError("failed to create CGL pixel format")

            self._context = cgl.CGLContextObj()
            cgl.CGLCreateContext(
                self._pixel_format,
                0,
                ctypes.byref(self._context),
            )
            if not self._context:
                cgl.CGLReleasePixelFormat(self._pixel_format)
                self._pixel_format = None
                raise RuntimeError("failed to create CGL context")

        def _choose_pixel_format(
            self, cgl: Any, attrib_values: tuple[int, ...]
        ) -> bool:
            attribs = (ctypes.c_int * len(attrib_values))(*attrib_values)
            pixel_format = cgl.CGLPixelFormatObj()
            num_pixel_formats = cgl.GLint()
            try:
                cgl.CGLChoosePixelFormat(
                    attribs,
                    ctypes.byref(pixel_format),
                    ctypes.byref(num_pixel_formats),
                )
            except cgl.CGLError:
                return False
            if not pixel_format or num_pixel_formats.value == 0:
                if pixel_format:
                    cgl.CGLReleasePixelFormat(pixel_format)
                return False
            self._pixel_format = pixel_format
            return True

        def make_current(self) -> None:
            from mujoco.cgl import cgl

            cgl.CGLSetCurrentContext(self._context)
            # Keep the lock lifecycle idempotent for repeated Renderer.render()
            # calls, while matching EnvPool's native CGL context setup.
            if not self._locked:
                cgl.CGLLockContext(self._context)
                self._locked = True

        def free(self) -> None:
            from mujoco.cgl import cgl

            if self._context:
                if self._locked:
                    cgl.CGLUnlockContext(self._context)
                    self._locked = False
                cgl.CGLSetCurrentContext(None)
                cgl.CGLReleaseContext(self._context)
                self._context = None
            if self._pixel_format:
                cgl.CGLReleasePixelFormat(self._pixel_format)
                self._pixel_format = None

        def __del__(self) -> None:
            try:
                self.free()
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    mujoco_gl_context.GLContext = _CglContext
    mujoco.GLContext = _CglContext
    try:
        import mujoco.cgl as mujoco_cgl

        mujoco_cgl.GLContext = _CglContext
    except (AttributeError, ImportError):
        pass
    try:
        from mujoco.rendering.classic import gl_context as classic_gl_context

        classic_gl_context.GLContext = _CglContext
    except (AttributeError, ImportError):
        pass
    try:
        from mujoco.rendering.classic import renderer as classic_renderer

        classic_renderer.gl_context.GLContext = _CglContext
    except (AttributeError, ImportError):
        pass


_configure_macos_mujoco_renderer()

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
_ROLLOUT_STEPS = 3
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
_STRICT_OBS_ATOL = 5e-11
_STRICT_OBS_RTOL = 5e-11
_STRICT_MJX_DIAGNOSTIC_ATOL = 1e-10
_STRICT_MUJOCO_STATE_ATOL = 1e-12
_LEAP_ARM64_QVEL_ATOL = 5e-12


def _is_arm64() -> bool:
    return platform.machine().lower() in ("aarch64", "arm64")


def _bazel_shard() -> tuple[int, int]:
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
        _find_runfile("mujoco_menagerie_playground").as_posix()
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


def _h1_contact_by_sensor_id(oracle: Any, data: mjx.Data) -> jax.Array:
    # Upstream H1 reset uses sensor_adr, but step indexes sensordata directly
    # by sensor id. Mirror that behavior exactly for oracle alignment.
    left = jnp.array([
        data.sensordata[sensor_id] > 0
        for sensor_id in oracle._left_foot_floor_found_sensor
    ])
    right = jnp.array([
        data.sensordata[sensor_id] > 0
        for sensor_id in oracle._right_foot_floor_found_sensor
    ])
    return jnp.hstack([jnp.any(left), jnp.any(right)])


def _step_mujoco_berkeley_oracle(
    oracle: oracle_berkeley.Joystick,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    info = dict(state.info)
    metrics = dict(state.metrics)
    info["rng"], push1_rng, push2_rng = jax.random.split(info["rng"], 3)
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jnp.pi)
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=oracle._config.push_config.magnitude_range[0],  # pylint: disable=protected-access
        maxval=oracle._config.push_config.magnitude_range[1],  # pylint: disable=protected-access
    )
    push = jnp.array([jnp.cos(push_theta), jnp.sin(push_theta)])
    push *= jnp.mod(info["push_step"] + 1, info["push_interval_steps"]) == 0
    push *= oracle._config.push_config.enable  # pylint: disable=protected-access
    data.qvel[:2] += np.asarray(push) * float(push_magnitude)

    motor_targets = (
        np.asarray(oracle._default_pose)  # pylint: disable=protected-access
        + action * oracle._config.action_scale  # pylint: disable=protected-access
    )
    data.ctrl[:] = motor_targets
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)

    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)
    contact = _berkeley_contact(oracle, mjx_data)
    contact_filt = contact | info["last_contact"]
    first_contact = (info["feet_air_time"] > 0.0) * contact_filt
    info["feet_air_time"] += oracle.dt
    foot_pos = mjx_data.site_xpos[oracle._feet_site_id]  # pylint: disable=protected-access
    info["swing_peak"] = jnp.maximum(info["swing_peak"], foot_pos[..., -1])

    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info, contact
    )
    done = oracle._get_termination(  # pylint: disable=protected-access
        mjx_data
    )
    rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data,
        jnp.asarray(action),
        info,
        metrics,
        done,
        first_contact,
        contact,
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in rewards.items()
    }
    reward = jnp.clip(sum(rewards.values()) * oracle.dt, 0.0, 10000.0)

    info["push"] = push
    info["step"] += 1
    info["push_step"] += 1
    phase_tp1 = info["phase"] + info["phase_dt"]
    info["phase"] = jnp.fmod(phase_tp1 + jnp.pi, 2 * jnp.pi) - jnp.pi
    info["last_last_act"] = info["last_act"]
    info["last_act"] = jnp.asarray(action)
    info["rng"], cmd_rng = jax.random.split(info["rng"])
    info["command"] = jnp.where(
        info["step"] > 500,
        oracle.sample_command(cmd_rng),
        info["command"],
    )
    info["step"] = jnp.where(done | (info["step"] > 500), 0, info["step"])
    info["feet_air_time"] *= ~contact
    info["last_contact"] = contact
    info["swing_peak"] *= ~contact
    for key, value in rewards.items():
        metrics[f"reward/{key}"] = value
    metrics["swing_peak"] = jnp.mean(info["swing_peak"])
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        info=info,
        metrics=metrics,
    )


def _step_mujoco_g1_oracle(
    oracle: oracle_g1.Joystick,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    info = dict(state.info)
    metrics = dict(state.metrics)
    info["rng"], push1_rng, push2_rng = jax.random.split(info["rng"], 3)
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jnp.pi)
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=oracle._config.push_config.magnitude_range[0],  # pylint: disable=protected-access
        maxval=oracle._config.push_config.magnitude_range[1],  # pylint: disable=protected-access
    )
    push = jnp.array([jnp.cos(push_theta), jnp.sin(push_theta)])
    push *= jnp.mod(info["push_step"] + 1, info["push_interval_steps"]) == 0
    push *= oracle._config.push_config.enable  # pylint: disable=protected-access
    data.qvel[:2] += np.asarray(push) * float(push_magnitude)

    motor_targets = (
        np.asarray(oracle._default_pose)  # pylint: disable=protected-access
        + action * oracle._config.action_scale  # pylint: disable=protected-access
    )
    data.ctrl[:] = motor_targets
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)

    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)
    contact = _g1_contact(oracle, mjx_data)
    contact_filt = contact | info["last_contact"]
    first_contact = (info["feet_air_time"] > 0.0) * contact_filt
    info["feet_air_time"] += oracle.dt
    foot_pos = mjx_data.site_xpos[oracle._feet_site_id]  # pylint: disable=protected-access
    info["swing_peak"] = jnp.maximum(info["swing_peak"], foot_pos[..., -1])

    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info, contact
    )
    done = oracle._get_termination(  # pylint: disable=protected-access
        mjx_data
    )
    rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data,
        jnp.asarray(action),
        info,
        metrics,
        done,
        first_contact,
        contact,
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in rewards.items()
    }
    reward = sum(rewards.values()) * oracle.dt

    info["push"] = push
    info["step"] += 1
    info["push_step"] += 1
    phase_tp1 = info["phase"] + info["phase_dt"]
    info["phase"] = jnp.fmod(phase_tp1 + jnp.pi, 2 * jnp.pi) - jnp.pi
    info["last_last_act"] = info["last_act"]
    info["last_act"] = jnp.asarray(action)
    info["rng"], cmd_rng = jax.random.split(info["rng"])
    info["command"] = jnp.where(
        info["step"] > 500,
        oracle.sample_command(cmd_rng),
        info["command"],
    )
    info["step"] = jnp.where(done | (info["step"] > 500), 0, info["step"])
    info["feet_air_time"] *= ~contact
    info["last_contact"] = contact
    info["swing_peak"] *= ~contact
    for key, value in rewards.items():
        metrics[f"reward/{key}"] = value
    metrics["swing_peak"] = jnp.mean(info["swing_peak"])
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        info=info,
        metrics=metrics,
    )


def _step_mujoco_t1_oracle(
    oracle: oracle_t1.Joystick,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    info = dict(state.info)
    metrics = dict(state.metrics)
    info["rng"], push1_rng, push2_rng = jax.random.split(info["rng"], 3)
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jnp.pi)
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=oracle._config.push_config.magnitude_range[0],  # pylint: disable=protected-access
        maxval=oracle._config.push_config.magnitude_range[1],  # pylint: disable=protected-access
    )
    push = jnp.array([jnp.cos(push_theta), jnp.sin(push_theta)])
    push *= jnp.mod(info["push_step"] + 1, info["push_interval_steps"]) == 0
    push *= oracle._config.push_config.enable  # pylint: disable=protected-access
    data.qvel[:2] += np.asarray(push) * float(push_magnitude)

    motor_targets = (
        np.asarray(oracle._default_pose)  # pylint: disable=protected-access
        + action * oracle._config.action_scale  # pylint: disable=protected-access
    )
    data.ctrl[:] = motor_targets
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)

    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)
    info["motor_targets"] = jnp.asarray(motor_targets)
    info["filtered_linvel"] = oracle.get_local_linvel(mjx_data)
    info["filtered_angvel"] = oracle.get_gyro(mjx_data)
    contact = _t1_contact(oracle, mjx_data)
    contact_filt = contact | info["last_contact"]
    first_contact = (info["feet_air_time"] > 0.0) * contact_filt
    info["feet_air_time"] += oracle.dt
    foot_pos = mjx_data.site_xpos[oracle._feet_site_id]  # pylint: disable=protected-access
    info["swing_peak"] = jnp.maximum(info["swing_peak"], foot_pos[..., -1])

    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info, contact
    )
    done = oracle._get_termination(  # pylint: disable=protected-access
        mjx_data
    )
    rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data,
        jnp.asarray(action),
        info,
        metrics,
        done,
        first_contact,
        contact,
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in rewards.items()
    }
    reward = jnp.clip(sum(rewards.values()) * oracle.dt, 0.0, 10000.0)

    info["push"] = push
    info["step"] += 1
    info["push_step"] += 1
    phase_tp1 = info["phase"] + info["phase_dt"]
    info["phase"] = jnp.fmod(phase_tp1 + jnp.pi, 2 * jnp.pi) - jnp.pi
    info["phase"] = jnp.where(
        jnp.linalg.norm(info["command"]) > 0.01,
        info["phase"],
        jnp.ones(2) * jnp.pi,
    )
    info["last_last_act"] = info["last_act"]
    info["last_act"] = jnp.asarray(action)
    info["rng"], cmd_rng = jax.random.split(info["rng"])
    info["command"] = jnp.where(
        info["step"] > 500,
        oracle.sample_command(cmd_rng),
        info["command"],
    )
    info["step"] = jnp.where(done | (info["step"] > 500), 0, info["step"])
    info["feet_air_time"] *= ~contact
    info["last_contact"] = contact
    info["swing_peak"] *= ~contact
    for key, value in rewards.items():
        metrics[f"reward/{key}"] = value
    metrics["swing_peak"] = jnp.mean(info["swing_peak"])
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        info=info,
        metrics=metrics,
    )


def _step_berkeley_oracle_from_envpool_info(
    oracle: oracle_berkeley.Joystick,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    info_after_step: dict[str, Any],
) -> oracle_mjx_env.State:
    """Apply official Berkeley formulas to EnvPool's native MuJoCo state.

    Berkeley rough terrain is contact-sensitive enough that the pip MuJoCo
    wheel and EnvPool's linked MuJoCo source build diverge after a single hfield
    solver step. The formulas are still official Playground code; only the
    physics state comes from EnvPool's native runtime.
    """
    mujoco_data = _make_mujoco_data_from_info(oracle, info_after_step)
    mjx_data = _sync_mjx_data_from_mujoco(
        state.data, mujoco_data, oracle.mj_model
    )
    if "site_xpos" in info_after_step and "site_xmat" in info_after_step:
        mjx_data = mjx_data.replace(
            site_xpos=jnp.asarray(
                info_after_step["site_xpos"][
                    0, : oracle.mj_model.nsite * 3
                ].reshape(oracle.mj_model.nsite, 3)
            ),
            site_xmat=jnp.asarray(
                info_after_step["site_xmat"][
                    0, : oracle.mj_model.nsite * 9
                ].reshape(oracle.mj_model.nsite, 3, 3)
            ),
        )
    if "actuator_force" in info_after_step:
        mjx_data = mjx_data.replace(
            actuator_force=jnp.asarray(info_after_step["actuator_force"][0])
        )
    info = dict(state.info)
    metrics = dict(state.metrics)
    contact = _berkeley_contact(oracle, mjx_data)
    contact_filt = contact | info["last_contact"]
    first_contact = (info["feet_air_time"] > 0.0) * contact_filt
    info["feet_air_time"] += oracle.dt
    foot_pos = mjx_data.site_xpos[oracle._feet_site_id]  # pylint: disable=protected-access
    info["swing_peak"] = jnp.maximum(info["swing_peak"], foot_pos[..., -1])

    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info, contact
    )
    done = oracle._get_termination(  # pylint: disable=protected-access
        mjx_data
    )
    rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data,
        jnp.asarray(action),
        info,
        metrics,
        done,
        first_contact,
        contact,
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in rewards.items()
    }
    reward = jnp.clip(sum(rewards.values()) * oracle.dt, 0.0, 10000.0)

    info["push"] = jnp.asarray(info_after_step["push"][0])
    info["step"] += 1
    info["push_step"] += 1
    phase_tp1 = info["phase"] + info["phase_dt"]
    info["phase"] = jnp.fmod(phase_tp1 + jnp.pi, 2 * jnp.pi) - jnp.pi
    info["last_last_act"] = info["last_act"]
    info["last_act"] = jnp.asarray(action)
    info["rng"], cmd_rng = jax.random.split(info["rng"])
    info["command"] = jnp.where(
        info["step"] > 500,
        oracle.sample_command(cmd_rng),
        info["command"],
    )
    info["step"] = jnp.where(done | (info["step"] > 500), 0, info["step"])
    info["feet_air_time"] *= ~contact
    info["last_contact"] = contact
    info["swing_peak"] *= ~contact
    for key, value in rewards.items():
        metrics[f"reward/{key}"] = value
    metrics["swing_peak"] = jnp.mean(info["swing_peak"])
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        info=info,
        metrics=metrics,
    )


def _step_g1_oracle_from_envpool_info(
    oracle: oracle_g1.Joystick,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    info_after_step: dict[str, Any],
) -> oracle_mjx_env.State:
    """Apply official G1 formulas to EnvPool's native MuJoCo state."""
    mujoco_data = _make_mujoco_data_from_info(oracle, info_after_step)
    mjx_data = _sync_mjx_data_from_mujoco(
        state.data, mujoco_data, oracle.mj_model
    )
    if "site_xpos" in info_after_step and "site_xmat" in info_after_step:
        mjx_data = mjx_data.replace(
            site_xpos=jnp.asarray(
                info_after_step["site_xpos"][
                    0, : oracle.mj_model.nsite * 3
                ].reshape(oracle.mj_model.nsite, 3)
            ),
            site_xmat=jnp.asarray(
                info_after_step["site_xmat"][
                    0, : oracle.mj_model.nsite * 9
                ].reshape(oracle.mj_model.nsite, 3, 3)
            ),
        )
    if "actuator_force" in info_after_step:
        mjx_data = mjx_data.replace(
            actuator_force=jnp.asarray(info_after_step["actuator_force"][0])
        )
    info = dict(state.info)
    metrics = dict(state.metrics)
    contact = _g1_contact(oracle, mjx_data)
    contact_filt = contact | info["last_contact"]
    first_contact = (info["feet_air_time"] > 0.0) * contact_filt
    info["feet_air_time"] += oracle.dt
    foot_pos = mjx_data.site_xpos[oracle._feet_site_id]  # pylint: disable=protected-access
    info["swing_peak"] = jnp.maximum(info["swing_peak"], foot_pos[..., -1])

    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info, contact
    )
    done = oracle._get_termination(  # pylint: disable=protected-access
        mjx_data
    )
    rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data,
        jnp.asarray(action),
        info,
        metrics,
        done,
        first_contact,
        contact,
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in rewards.items()
    }
    reward = sum(rewards.values()) * oracle.dt

    info["push"] = jnp.asarray(info_after_step["push"][0])
    info["step"] += 1
    info["push_step"] += 1
    phase_tp1 = info["phase"] + info["phase_dt"]
    info["phase"] = jnp.fmod(phase_tp1 + jnp.pi, 2 * jnp.pi) - jnp.pi
    info["last_last_act"] = info["last_act"]
    info["last_act"] = jnp.asarray(action)
    info["rng"], cmd_rng = jax.random.split(info["rng"])
    info["command"] = jnp.where(
        info["step"] > 500,
        oracle.sample_command(cmd_rng),
        info["command"],
    )
    info["step"] = jnp.where(done | (info["step"] > 500), 0, info["step"])
    info["feet_air_time"] *= ~contact
    info["last_contact"] = contact
    info["swing_peak"] *= ~contact
    for key, value in rewards.items():
        metrics[f"reward/{key}"] = value
    metrics["swing_peak"] = jnp.mean(info["swing_peak"])
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        info=info,
        metrics=metrics,
    )


def _step_panda_cartesian_oracle_from_envpool_info(
    oracle: oracle_panda_pick_cartesian.PandaPickCubeCartesian,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    info_after_step: dict[str, Any],
) -> oracle_mjx_env.State:
    mujoco_data = _make_mujoco_data_from_info(oracle, info_after_step)
    mjx_data = _sync_mjx_data_from_mujoco(
        state.data, mujoco_data, oracle.mj_model
    )
    mjx_data = mjx_data.replace(
        xpos=jnp.asarray(
            info_after_step["xpos"][0, : oracle.mj_model.nbody * 3].reshape(
                oracle.mj_model.nbody, 3
            )
        ),
        xmat=jnp.asarray(
            info_after_step["xmat"][0, : oracle.mj_model.nbody * 9].reshape(
                oracle.mj_model.nbody, 3, 3
            )
        ),
        site_xpos=jnp.asarray(
            info_after_step["site_xpos"][
                0, : oracle.mj_model.nsite * 3
            ].reshape(oracle.mj_model.nsite, 3)
        ),
        site_xmat=jnp.asarray(
            info_after_step["site_xmat"][
                0, : oracle.mj_model.nsite * 9
            ].reshape(oracle.mj_model.nsite, 3, 3)
        ),
    )
    info = dict(state.info)
    metrics = dict(state.metrics)
    newly_reset = info["_steps"] == 0
    raw_rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data, info
    )
    rewards = {
        key: value * oracle._config.reward_config.reward_scales[key]  # pylint: disable=protected-access
        for key, value in raw_rewards.items()
    }
    hand_box = (
        mjx_data.sensordata[
            oracle._mj_model.sensor_adr[  # pylint: disable=protected-access
                oracle._box_hand_found_sensor  # pylint: disable=protected-access
            ]
        ]
        > 0
    )
    raw_rewards["no_box_collision"] = jnp.where(hand_box, 0.0, 1.0)
    total_reward = jnp.clip(sum(rewards.values()), -1e4, 1e4)
    total_reward += oracle._config.reward_config.action_rate * jnp.linalg.norm(  # pylint: disable=protected-access
        jnp.asarray(action) - info["prev_action"]
    )
    total_reward += (
        jnp.asarray(info_after_step["no_soln"][0])
        * oracle._config.reward_config.no_soln_reward  # pylint: disable=protected-access
    )
    box_pos = mjx_data.xpos[oracle._obj_body]  # pylint: disable=protected-access
    lifted = (box_pos[2] > 0.05) * oracle._config.reward_config.lifted_reward  # pylint: disable=protected-access
    total_reward += lifted
    success = oracle._get_success(  # pylint: disable=protected-access
        mjx_data, info
    )
    total_reward += success * oracle._config.reward_config.success_reward  # pylint: disable=protected-access
    reward = jnp.maximum(total_reward - info["prev_reward"], 0.0)
    reward = jnp.where(newly_reset, 0.0, reward)

    for key in (
        "target_pos",
        "reached_box",
        "prev_reward",
        "current_pos",
        "prev_action",
        "no_soln",
    ):
        info[key] = jnp.asarray(info_after_step[key][0])
    info["_steps"] = jnp.asarray(info_after_step["steps"][0], dtype=jnp.int32)
    metrics.update({
        f"reward/{key}": value for key, value in raw_rewards.items()
    })
    metrics.update({
        "out_of_bounds": jnp.asarray(info_after_step["out_of_bounds"][0]),
        "reward/lifted": lifted.astype(float),
        "reward/success": success.astype(float),
    })
    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info
    )
    obs = jnp.concat(
        [obs, jnp.reshape(info["no_soln"], (1,)), info["prev_action"]],
        axis=0,
    )
    done = jnp.asarray(info_after_step["terminated"][0]).astype(float)
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done,
        info=info,
        metrics=metrics,
    )


def _step_panda_robotiq_oracle_from_envpool_info(
    oracle: oracle_panda_robotiq.PandaRobotiqPushCube,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    info_after_step: dict[str, Any],
) -> oracle_mjx_env.State:
    mujoco_data = _make_mujoco_data_from_info(oracle, info_after_step)
    mjx_data = _sync_mjx_data_from_mujoco(
        state.data, mujoco_data, oracle.mj_model
    )
    mjx_data = mjx_data.replace(
        xpos=jnp.asarray(
            info_after_step["xpos"][0, : oracle.mj_model.nbody * 3].reshape(
                oracle.mj_model.nbody, 3
            )
        ),
        xquat=jnp.asarray(
            info_after_step["xquat"][0, : oracle.mj_model.nbody * 4].reshape(
                oracle.mj_model.nbody, 4
            )
        ),
        xmat=jnp.asarray(
            info_after_step["xmat"][0, : oracle.mj_model.nbody * 9].reshape(
                oracle.mj_model.nbody, 3, 3
            )
        ),
        site_xpos=jnp.asarray(
            info_after_step["site_xpos"][
                0, : oracle.mj_model.nsite * 3
            ].reshape(oracle.mj_model.nsite, 3)
        ),
        site_xmat=jnp.asarray(
            info_after_step["site_xmat"][
                0, : oracle.mj_model.nsite * 9
            ].reshape(oracle.mj_model.nsite, 3, 3)
        ),
        sensordata=jnp.asarray(
            info_after_step["sensordata"][0, : mjx_data.sensordata.size]
        ),
    )

    info = dict(state.info)
    metrics = dict(state.metrics)
    rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data, info, jnp.asarray(action)
    )
    rewards = {
        key: value * oracle._config.reward_config.reward_scales[key]  # pylint: disable=protected-access
        for key, value in rewards.items()
    }
    reward = jnp.clip(sum(rewards.values()), -1e4, 1e4)
    reward_scale_sum = sum(
        oracle._config.reward_config.reward_scales[key]  # pylint: disable=protected-access
        for key in rewards
    )
    reward /= reward_scale_sum
    termination = oracle._get_termination(  # pylint: disable=protected-access
        mjx_data
    )
    reward += oracle._config.reward_config.termination_reward * termination  # pylint: disable=protected-access
    working_state = state.replace(data=mjx_data, info=info, metrics=metrics)
    working_state, success, success_wait = oracle._get_success(  # pylint: disable=protected-access
        working_state
    )
    metrics = dict(working_state.metrics)
    reward += (
        oracle._config.reward_config.success_wait_reward * success_wait  # pylint: disable=protected-access
    )
    reward += oracle._config.reward_config.success_reward * success  # pylint: disable=protected-access
    reward *= oracle.dt

    obs = oracle._get_obs(working_state)  # pylint: disable=protected-access
    info = dict(working_state.info)
    info["last_action"] = jnp.asarray(info_after_step["last_action"][0])
    action_history_size = int(oracle._config.action_history_len) * 7  # pylint: disable=protected-access
    obs_history_size = int(oracle._config.obs_history_len) * 48  # pylint: disable=protected-access
    info["action_history"] = jnp.asarray(
        info_after_step["action_history"][0, :action_history_size]
    )
    info["obs_history"] = jnp.asarray(
        info_after_step["obs_history"][0, :obs_history_size]
    )
    for key in ("success_step_count", "prev_step_success", "curriculum_id"):
        info[key] = jnp.asarray(info_after_step[key][0], dtype=jnp.int32)
    info["rng"], _ = jax.random.split(info["rng"])
    metrics.update(out_of_bounds=termination.astype(float), **rewards)
    done = (
        termination
        | jnp.isnan(mjx_data.qpos).any()
        | jnp.isnan(mjx_data.qvel).any()
    )
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(float),
        info=info,
        metrics=metrics,
    )


def _step_mujoco_hand_rotate_oracle(
    oracle: Any,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    info = dict(state.info)
    metrics = dict(state.metrics)
    if isinstance(oracle, oracle_aero_rotate.CubeRotateZAxis):
        motor_targets = np.asarray(
            oracle._default_tendon
        ) + action * np.asarray(  # pylint: disable=protected-access
            oracle._config.action_scale  # pylint: disable=protected-access
        )
    else:
        motor_targets = (
            np.asarray(oracle._default_pose)  # pylint: disable=protected-access
            + action * oracle._config.action_scale  # pylint: disable=protected-access
        )
    data.ctrl[:] = motor_targets
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)

    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)
    info["motor_targets"] = jnp.asarray(motor_targets)
    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info, state.obs["state"]
    )
    done = oracle._get_termination(  # pylint: disable=protected-access
        mjx_data
    )
    rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data, jnp.asarray(action), info, metrics, done
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in rewards.items()
    }
    reward = sum(rewards.values()) * oracle.dt
    info["last_last_act"] = info["last_act"]
    info["last_act"] = jnp.asarray(action)
    info["last_cube_angvel"] = oracle.get_cube_angvel(mjx_data)
    for key, value in rewards.items():
        metrics[f"reward/{key}"] = value
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        info=info,
        metrics=metrics,
    )


def _step_mujoco_leap_reorient_oracle(
    oracle: oracle_leap_reorient.CubeReorient,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    info = dict(state.info)
    metrics = dict(state.metrics)
    delta = action * oracle._config.action_scale  # pylint: disable=protected-access
    motor_targets = np.asarray(data.ctrl) + delta
    motor_targets = np.clip(
        motor_targets,
        np.asarray(oracle._lowers),  # pylint: disable=protected-access
        np.asarray(oracle._uppers),  # pylint: disable=protected-access
    )
    motor_targets = (
        oracle._config.ema_alpha * motor_targets  # pylint: disable=protected-access
        + (1 - oracle._config.ema_alpha) * np.asarray(info["motor_targets"])  # pylint: disable=protected-access
    )
    data.ctrl[:] = motor_targets
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)

    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)
    info["motor_targets"] = jnp.asarray(motor_targets)
    ori_error = oracle._cube_orientation_error(  # pylint: disable=protected-access
        mjx_data
    )
    success = ori_error < oracle._config.success_threshold  # pylint: disable=protected-access
    info["steps_since_last_success"] = jnp.where(
        success, 0, info["steps_since_last_success"] + 1
    )
    info["success_count"] = jnp.where(
        success, info["success_count"] + 1, info["success_count"]
    )
    metrics["steps_since_last_success"] = info["steps_since_last_success"]
    metrics["success_count"] = info["success_count"]

    done = oracle._get_termination(  # pylint: disable=protected-access
        mjx_data, info
    )
    obs = oracle._get_obs(mjx_data, info)  # pylint: disable=protected-access
    rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data, jnp.asarray(action), info, metrics, done
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in rewards.items()
    }
    reward = sum(rewards.values()) * oracle.dt

    info["rng"], goal_rng = jax.random.split(info["rng"])
    info["goal_quat_dquat"] = jnp.where(
        success,
        3 + jax.random.uniform(goal_rng, (3,), minval=-2, maxval=2),
        info["goal_quat_dquat"] * 0.8,
    )
    goal_quat = mjx_math.quat_integrate(
        state.data.mocap_quat[0],
        info["goal_quat_dquat"],
        2 * jnp.array(oracle.dt),
    )
    mjx_data = mjx_data.replace(mocap_quat=jnp.array([goal_quat]))
    data.mocap_quat[:] = np.asarray(mjx_data.mocap_quat)
    metrics["reward/success"] = success.astype(float)
    reward += success * oracle._config.reward_config.success_reward  # pylint: disable=protected-access

    info["step"] += 1
    info["last_last_act"] = info["last_act"]
    info["last_act"] = jnp.asarray(action)
    for key, value in rewards.items():
        metrics[f"reward/{key}"] = value
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        info=info,
        metrics=metrics,
    )


def _step_oracle_from_envpool_info(
    oracle: Any,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    info_after_step: dict[str, Any],
) -> oracle_mjx_env.State:
    if isinstance(oracle, oracle_berkeley.Joystick):
        return _step_berkeley_oracle_from_envpool_info(
            oracle, state, action, info_after_step
        )
    if isinstance(oracle, oracle_g1.Joystick):
        return _step_g1_oracle_from_envpool_info(
            oracle, state, action, info_after_step
        )
    if isinstance(oracle, oracle_panda_pick_cartesian.PandaPickCubeCartesian):
        return _step_panda_cartesian_oracle_from_envpool_info(
            oracle, state, action, info_after_step
        )
    if isinstance(oracle, oracle_panda_robotiq.PandaRobotiqPushCube):
        return _step_panda_robotiq_oracle_from_envpool_info(
            oracle, state, action, info_after_step
        )
    raise TypeError(f"Unsupported native-state oracle {type(oracle)!r}")


def _step_mujoco_joystick_oracle(
    oracle: oracle_go1.Joystick,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    motor_targets = (
        np.asarray(oracle._default_pose)  # pylint: disable=protected-access
        + action * oracle._config.action_scale  # pylint: disable=protected-access
    )
    data.ctrl[:] = motor_targets
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)

    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)
    info = dict(state.info)
    metrics = dict(state.metrics)
    contact = jnp.array([
        mjx_data.sensordata[oracle._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in oracle._feet_floor_found_sensor
    ])
    contact_filt = contact | info["last_contact"]
    first_contact = (info["feet_air_time"] > 0.0) * contact_filt
    info["feet_air_time"] = info["feet_air_time"] + oracle.dt
    foot_pos = mjx_data.site_xpos[oracle._feet_site_id]
    info["swing_peak"] = jnp.maximum(info["swing_peak"], foot_pos[..., -1])

    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info
    )
    done = oracle._get_termination(  # pylint: disable=protected-access
        mjx_data
    )
    rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data,
        jnp.asarray(action),
        info,
        metrics,
        done,
        first_contact,
        contact,
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in rewards.items()
    }
    reward = jnp.clip(sum(rewards.values()) * oracle.dt, 0.0, 10000.0)

    info["last_last_act"] = info["last_act"]
    info["last_act"] = jnp.asarray(action)
    info["steps_until_next_cmd"] = info["steps_until_next_cmd"] - 1
    info["rng"], key1, key2 = jax.random.split(info["rng"], 3)
    info["command"] = jnp.where(
        info["steps_until_next_cmd"] <= 0,
        oracle.sample_command(key1, info["command"]),
        info["command"],
    )
    info["steps_until_next_cmd"] = jnp.where(
        done | (info["steps_until_next_cmd"] <= 0),
        jnp.round(jax.random.exponential(key2) * 5.0 / oracle.dt).astype(
            jnp.int32
        ),
        info["steps_until_next_cmd"],
    )
    info["feet_air_time"] = info["feet_air_time"] * ~contact
    info["last_contact"] = contact
    info["swing_peak"] = info["swing_peak"] * ~contact
    for key, value in rewards.items():
        metrics[f"reward/{key}"] = value
    metrics["swing_peak"] = jnp.mean(info["swing_peak"])
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        info=info,
        metrics=metrics,
    )


def _step_mujoco_apollo_oracle(
    oracle: oracle_apollo.Joystick,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    motor_targets = (
        np.asarray(oracle._default_ctrl)  # pylint: disable=protected-access
        + action * oracle._config.action_scale  # pylint: disable=protected-access
    )
    data.ctrl[:] = motor_targets
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)

    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)
    info = dict(state.info)
    metrics = dict(state.metrics)
    linvel = oracle.get_local_linvel(mjx_data)
    info["filtered_linvel"] = linvel * 1.0 + info["filtered_linvel"] * 0.0
    angvel = oracle.get_gyro(mjx_data)
    info["filtered_angvel"] = angvel * 1.0 + info["filtered_angvel"] * 0.0

    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info
    )
    done = oracle._get_termination(  # pylint: disable=protected-access
        mjx_data, metrics
    )
    rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data,
        jnp.asarray(action),
        info,
        metrics,
        done,
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in rewards.items()
    }
    reward = sum(rewards.values()) * oracle.dt

    info["step"] += 1
    phase_tp1 = info["phase"] + info["phase_dt"]
    info["phase"] = jnp.fmod(phase_tp1 + jnp.pi, 2 * jnp.pi) - jnp.pi
    info["phase"] = jnp.where(
        jnp.linalg.norm(info["command"]) > 0.01,
        info["phase"],
        jnp.ones(2) * jnp.pi,
    )
    info["last_act"] = jnp.asarray(action)
    info["steps_until_next_cmd"] -= 1
    info["rng"], key1, key2 = jax.random.split(info["rng"], 3)
    info["command"] = jnp.where(
        info["steps_until_next_cmd"] <= 0,
        oracle.sample_command(key1, info["command"]),
        info["command"],
    )
    info["steps_until_next_cmd"] = jnp.where(
        done | (info["steps_until_next_cmd"] <= 0),
        jnp.round(jax.random.exponential(key2) * 5.0 / oracle.dt).astype(
            jnp.int32
        ),
        info["steps_until_next_cmd"],
    )
    for key, value in rewards.items():
        metrics[f"reward/{key}"] = value
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        info=info,
        metrics=metrics,
    )


def _step_mujoco_barkour_oracle(
    oracle: oracle_barkour.Joystick,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    data.xfrc_applied[:] = np.asarray(state.data.xfrc_applied)
    motor_targets = (
        np.asarray(oracle._default_pose)  # pylint: disable=protected-access
        + action * oracle._config.action_scale  # pylint: disable=protected-access
    )
    motor_targets = np.clip(
        motor_targets,
        np.asarray(oracle._lowers),  # pylint: disable=protected-access
        np.asarray(oracle._uppers),  # pylint: disable=protected-access
    )
    data.ctrl[:] = motor_targets
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)

    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)
    info = dict(state.info)
    metrics = dict(state.metrics)
    rng, cmd_rng, noise_rng, _ = jax.random.split(info["rng"], 4)
    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info, state.obs, noise_rng
    )
    joint_angles = mjx_data.qpos[7:]
    joint_vel = mjx_data.qvel[6:]
    torso_z = mjx_data.xpos[oracle._torso_body_id, -1]  # pylint: disable=protected-access
    contact = jnp.array([
        mjx_data.sensordata[
            oracle._mj_model.sensor_adr[  # pylint: disable=protected-access
                oracle._mj_model.sensor(sensor).id  # pylint: disable=protected-access
            ]
        ]
        > 0
        for sensor in oracle._feet_floor_found_sensor  # pylint: disable=protected-access
    ])
    contact_filt = contact | info["last_contact"]
    first_contact = (info["feet_air_time"] > 0.0) * contact_filt
    info["feet_air_time"] += oracle.dt

    done = oracle._get_gravity(mjx_data)[-1] < 0  # pylint: disable=protected-access
    done |= jnp.any(joint_angles < oracle._lowers)  # pylint: disable=protected-access
    done |= jnp.any(joint_angles > oracle._uppers)  # pylint: disable=protected-access
    done |= torso_z < 0.18

    rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data, jnp.asarray(action), info, metrics, done, first_contact
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in rewards.items()
    }
    reward = jnp.clip(sum(rewards.values()) * oracle.dt, 0.0, 10000.0)

    info["last_act"] = jnp.asarray(action)
    info["last_vel"] = joint_vel
    info["step"] += 1
    info["rng"] = rng
    info["feet_air_time"] *= ~contact
    info["last_contact"] = contact
    info["command"] = jnp.where(
        info["step"] > 500,
        oracle.sample_command(cmd_rng),
        info["command"],
    )
    info["step"] = jnp.where(done | (info["step"] > 500), 0, info["step"])
    for key, value in rewards.items():
        metrics[f"reward/{key}"] = value
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        info=info,
        metrics=metrics,
    )


def _step_mujoco_op3_oracle(
    oracle: oracle_op3.Joystick,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    motor_targets = (
        np.asarray(oracle._default_pose)  # pylint: disable=protected-access
        + action * oracle._config.action_scale  # pylint: disable=protected-access
    )
    motor_targets = np.clip(
        motor_targets,
        np.asarray(oracle._lowers),  # pylint: disable=protected-access
        np.asarray(oracle._uppers),  # pylint: disable=protected-access
    )
    data.ctrl[:] = motor_targets
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)

    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)
    info = dict(state.info)
    metrics = dict(state.metrics)
    rng, cmd_rng, noise_rng = jax.random.split(info["rng"], 3)
    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info, state.obs, noise_rng
    )
    done = oracle._get_termination(  # pylint: disable=protected-access
        mjx_data
    )
    rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data, jnp.asarray(action), info, metrics, done
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in rewards.items()
    }
    reward = jnp.clip(sum(rewards.values()) * oracle.dt, 0.0, 10000.0)

    info["motor_targets"] = jnp.asarray(motor_targets)
    info["last_last_act"] = info["last_act"]
    info["last_act"] = jnp.asarray(action)
    info["last_vel"] = mjx_data.qvel[6:]
    info["step"] += 1
    info["rng"] = rng
    info["command"] = jnp.where(
        info["step"] > 500,
        oracle.sample_command(cmd_rng),
        info["command"],
    )
    info["step"] = jnp.where(done | (info["step"] > 500), 0, info["step"])
    for key, value in rewards.items():
        metrics[f"reward/{key}"] = value
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        info=info,
        metrics=metrics,
    )


def _step_mujoco_h1_inplace_oracle(
    oracle: oracle_h1_inplace.InplaceGaitTracking,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    info = dict(state.info)
    metrics = dict(state.metrics)
    rng, noise_rng = jax.random.split(info["rng"])
    motor_targets = (
        np.asarray(oracle._default_pose)  # pylint: disable=protected-access
        + action * oracle._config.action_scale  # pylint: disable=protected-access
    )
    motor_targets = np.clip(
        motor_targets,
        np.asarray(oracle._lowers),  # pylint: disable=protected-access
        np.asarray(oracle._uppers),  # pylint: disable=protected-access
    )
    data.ctrl[:] = motor_targets
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)

    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)
    info["motor_targets"] = jnp.asarray(motor_targets)
    contact = _h1_contact_by_sensor_id(oracle, mjx_data)
    foot_pos = mjx_data.site_xpos[oracle._feet_site_id]  # pylint: disable=protected-access
    info["swing_peak"] = jnp.maximum(info["swing_peak"], foot_pos[..., -1])

    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info, noise_rng, contact
    )
    done = oracle._get_termination(  # pylint: disable=protected-access
        mjx_data
    )
    pos, neg = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data, jnp.asarray(action), info, metrics, done
    )
    pos = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in pos.items()
    }
    neg = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in neg.items()
    }
    rewards = pos | neg
    reward = sum(pos.values()) * jnp.exp(0.2 * sum(neg.values())) * oracle.dt

    info["last_last_act"] = info["last_act"]
    info["last_act"] = jnp.asarray(action)
    phase_tp1 = info["phase"] + info["phase_dt"]
    info["phase"] = jnp.fmod(phase_tp1 + jnp.pi, 2 * jnp.pi) - jnp.pi
    info["rng"] = rng
    info["swing_peak"] *= ~contact
    info["left_contact"] = contact[0]
    info["right_contact"] = contact[1]
    for key, value in rewards.items():
        metrics[f"reward/{key}"] = value
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        info=info,
        metrics=metrics,
    )


def _step_mujoco_h1_joystick_oracle(
    oracle: oracle_h1_joystick.JoystickGaitTracking,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    info = dict(state.info)
    metrics = dict(state.metrics)
    rng, cmd_rng, noise_rng = jax.random.split(info["rng"], 3)
    motor_targets = (
        np.asarray(data.ctrl) + action * oracle._config.action_scale  # pylint: disable=protected-access
    )
    motor_targets = np.clip(
        motor_targets,
        np.asarray(oracle._lowers),  # pylint: disable=protected-access
        np.asarray(oracle._uppers),  # pylint: disable=protected-access
    )
    data.ctrl[:] = motor_targets
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)

    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)
    info["motor_targets"] = jnp.asarray(motor_targets)
    contact = _h1_contact_by_sensor_id(oracle, mjx_data)
    contact_filt = contact | info["last_contact"]
    first_contact = (info["feet_air_time"] > 0.0) * contact_filt
    info["feet_air_time"] += oracle.dt
    foot_pos = mjx_data.site_xpos[oracle._feet_site_id]  # pylint: disable=protected-access
    info["swing_peak"] = jnp.maximum(info["swing_peak"], foot_pos[..., -1])

    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info, noise_rng, contact
    )
    done = oracle._get_termination(  # pylint: disable=protected-access
        mjx_data
    )
    pos, neg = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data,
        jnp.asarray(action),
        info,
        metrics,
        done,
        first_contact,
        contact,
    )
    pos = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in pos.items()
    }
    neg = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in neg.items()
    }
    rewards = pos | neg
    reward = jnp.clip(sum(rewards.values()) * oracle.dt, 0.0)

    info["last_last_act"] = info["last_act"]
    info["last_act"] = jnp.asarray(action)
    info["step"] += 1
    phase_tp1 = info["phase"] + info["phase_dt"]
    info["phase"] = jnp.fmod(phase_tp1 + jnp.pi, 2 * jnp.pi) - jnp.pi
    info["rng"] = rng
    info["feet_air_time"] *= ~contact
    info["last_contact"] = contact
    info["swing_peak"] *= ~contact
    info["command"] = jnp.where(
        info["step"] > 500,
        oracle.sample_command(cmd_rng),
        info["command"],
    )
    info["step"] = jnp.where(done | (info["step"] > 500), 0, info["step"])
    for key, value in rewards.items():
        metrics[f"reward/{key}"] = value
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        info=info,
        metrics=metrics,
    )


def _step_mujoco_spot_joystick_oracle(
    oracle: oracle_spot_joystick.Joystick,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    info = dict(state.info)
    metrics = dict(state.metrics)
    rng, cmd_rng, noise_rng, _ = jax.random.split(info["rng"], 4)
    motor_targets = (
        np.asarray(oracle._default_pose)  # pylint: disable=protected-access
        + action * oracle._config.action_scale  # pylint: disable=protected-access
    )
    motor_targets = np.clip(
        motor_targets,
        np.asarray(oracle._lowers),  # pylint: disable=protected-access
        np.asarray(oracle._uppers),  # pylint: disable=protected-access
    )
    data.ctrl[:] = motor_targets
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)

    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)
    info["motor_targets"] = jnp.asarray(motor_targets)
    contact = _spot_contact(oracle, mjx_data)
    contact_filt = contact | info["last_contact"]
    first_contact = (info["feet_air_time"] > 0.0) * contact_filt
    info["feet_air_time"] += oracle.dt
    foot_pos = mjx_data.site_xpos[oracle._feet_site_id]  # pylint: disable=protected-access
    info["swing_peak"] = jnp.maximum(info["swing_peak"], foot_pos[..., -1])

    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info, noise_rng
    )
    done = oracle._get_termination(  # pylint: disable=protected-access
        mjx_data
    )
    rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data,
        jnp.asarray(action),
        info,
        metrics,
        done,
        first_contact,
        contact,
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in rewards.items()
    }
    reward = jnp.clip(sum(rewards.values()) * oracle.dt, 0.0, 10000.0)

    info["last_last_act"] = info["last_act"]
    info["last_act"] = jnp.asarray(action)
    info["step"] += 1
    info["rng"] = rng
    info["command"] = jnp.where(
        info["step"] > 200,
        oracle.sample_command(cmd_rng),
        info["command"],
    )
    info["step"] = jnp.where(done | (info["step"] > 200), 0, info["step"])
    info["feet_air_time"] *= ~contact
    info["last_contact"] = contact
    info["swing_peak"] *= ~contact
    for key, value in rewards.items():
        metrics[f"reward/{key}"] = value
    metrics["swing_peak"] = jnp.mean(info["swing_peak"])
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        info=info,
        metrics=metrics,
    )


def _step_mujoco_spot_getup_oracle(
    oracle: oracle_spot_getup.Getup,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    info = dict(state.info)
    metrics = dict(state.metrics)
    rng, noise_rng = jax.random.split(info["rng"], 2)
    motor_targets = (
        np.asarray(data.qpos[7:]) + action * oracle._config.action_scale  # pylint: disable=protected-access
    )
    motor_targets = np.clip(
        motor_targets,
        np.asarray(oracle._lowers),  # pylint: disable=protected-access
        np.asarray(oracle._uppers),  # pylint: disable=protected-access
    )
    data.ctrl[:] = motor_targets
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)

    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)
    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info, noise_rng
    )
    joint_angles = mjx_data.qpos[7:]
    done = jnp.any(joint_angles < oracle._lowers)  # pylint: disable=protected-access
    done |= jnp.any(joint_angles > oracle._uppers)  # pylint: disable=protected-access
    rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data,
        jnp.asarray(action),
        info,
        metrics,
        done,
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in rewards.items()
    }
    reward = jnp.clip(sum(rewards.values()) * oracle.dt, 0.0, 10000.0)

    info["last_last_act"] = info["last_act"]
    info["last_act"] = jnp.asarray(action)
    info["rng"] = rng
    for key, value in rewards.items():
        metrics[f"reward/{key}"] = value
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        info=info,
        metrics=metrics,
    )


def _step_mujoco_spot_gait_oracle(
    oracle: oracle_spot_gait.JoystickGaitTracking,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    info = dict(state.info)
    metrics = dict(state.metrics)
    rng, cmd_rng, noise_rng = jax.random.split(info["rng"], 3)
    motor_targets = (
        np.asarray(oracle._default_pose)  # pylint: disable=protected-access
        + action * oracle._config.action_scale  # pylint: disable=protected-access
    )
    motor_targets = np.clip(
        motor_targets,
        np.asarray(oracle._lowers),  # pylint: disable=protected-access
        np.asarray(oracle._uppers),  # pylint: disable=protected-access
    )
    data.ctrl[:] = motor_targets
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)

    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)
    info["motor_targets"] = jnp.asarray(motor_targets)
    contact = _spot_contact(oracle, mjx_data)
    foot_pos = mjx_data.site_xpos[oracle._feet_site_id]  # pylint: disable=protected-access
    info["swing_peak"] = jnp.maximum(info["swing_peak"], foot_pos[..., -1])

    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info, noise_rng, contact
    )
    done = oracle._get_termination(  # pylint: disable=protected-access
        mjx_data
    )
    pos, neg = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data,
        jnp.asarray(action),
        info,
        metrics,
        done,
    )
    pos = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in pos.items()
    }
    neg = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in neg.items()
    }
    rewards = pos | neg
    reward = sum(pos.values()) * jnp.exp(0.2 * sum(neg.values())) * oracle.dt

    info["last_last_act"] = info["last_act"]
    info["last_act"] = jnp.asarray(action)
    info["step"] += 1
    phase_tp1 = info["phase"] + info["phase_dt"]
    info["phase"] = jnp.fmod(phase_tp1 + jnp.pi, 2 * jnp.pi) - jnp.pi
    info["rng"] = rng
    info["command"] = jnp.where(
        info["step"] > 200,
        oracle.sample_command(cmd_rng),
        info["command"],
    )
    info["step"] = jnp.where(done | (info["step"] > 200), 0, info["step"])
    info["last_contact"] = contact
    info["swing_peak"] *= ~contact
    for key, value in rewards.items():
        metrics[f"reward/{key}"] = value
    metrics["swing_peak"] = jnp.mean(info["swing_peak"])
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        info=info,
        metrics=metrics,
    )


def _step_mujoco_getup_oracle(
    oracle: oracle_go1_getup.Getup,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    motor_targets = (
        np.asarray(state.data.qpos[7:]) + action * oracle._config.action_scale  # pylint: disable=protected-access
    )
    data.ctrl[:] = motor_targets
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)
    mujoco.mj_forward(oracle.mj_model, data)

    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)
    info = dict(state.info)
    metrics = dict(state.metrics)
    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info
    )
    done = oracle._get_termination(  # pylint: disable=protected-access
        mjx_data
    )
    rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data,
        jnp.asarray(action),
        info,
        metrics,
        done,
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in rewards.items()
    }
    reward = jnp.clip(sum(rewards.values()) * oracle.dt, 0.0, 10000.0)

    info["last_last_act"] = info["last_act"]
    info["last_act"] = jnp.asarray(action)
    for key, value in rewards.items():
        metrics[f"reward/{key}"] = value
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        info=info,
        metrics=metrics,
    )


def _step_mujoco_handstand_oracle(
    oracle: oracle_go1_handstand.Handstand,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    data.ctrl[:] = (
        data.ctrl + action * oracle._config.action_scale  # pylint: disable=protected-access
    )
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)
    mujoco.mj_forward(oracle.mj_model, data)

    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)
    info = dict(state.info)
    metrics = dict(state.metrics)
    contact = _handstand_unwanted_contact(oracle, mjx_data)
    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info, contact
    )
    done = oracle._get_termination(  # pylint: disable=protected-access
        mjx_data, info, contact
    )
    rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data,
        jnp.asarray(action),
        info,
        done,
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in rewards.items()
    }
    reward = jnp.clip(sum(rewards.values()) * oracle.dt, 0.0, 10000.0)

    info["last_act"] = jnp.asarray(action)
    for key, value in rewards.items():
        metrics[f"reward/{key}"] = value
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        info=info,
        metrics=metrics,
    )


def _step_mujoco_aloha_handover_oracle(
    oracle: oracle_aloha_handover.HandOver,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    info = dict(state.info)
    newly_reset = info["_steps"] == 0
    info["episode_picked"] = jnp.where(newly_reset, 0, info["episode_picked"])
    info["prev_potential"] = jnp.where(newly_reset, 0.0, info["prev_potential"])

    delta = action * oracle._config.action_scale  # pylint: disable=protected-access
    ctrl = np.asarray(data.ctrl) + delta
    ctrl = np.clip(
        ctrl,
        np.asarray(oracle._lowers),  # pylint: disable=protected-access
        np.asarray(oracle._uppers),  # pylint: disable=protected-access
    )
    data.ctrl[:] = ctrl
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)
    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)

    raw_rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data, info
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in raw_rewards.items()
    }
    potential = sum(rewards.values()) / sum(
        oracle._config.reward_config.scales.values()  # pylint: disable=protected-access
    )
    reward = jnp.maximum(
        potential - info["prev_potential"], jnp.zeros_like(potential)
    )
    box_pos = mjx_data.xpos[oracle._box_body]  # pylint: disable=protected-access
    left_gripper = mjx_data.site_xpos[oracle._left_gripper_site]  # pylint: disable=protected-access
    condition = oracle_aloha_handover.logistic_barrier(
        left_gripper[0], direction=-1
    ) * oracle_aloha_handover.logistic_barrier(box_pos[0], 0.10)
    reward += 0.02 * potential * condition
    info["prev_potential"] = jnp.maximum(potential, info["prev_potential"])
    reward = jnp.where(newly_reset, 0.0, reward)

    picked = box_pos[2] > 0.15
    info["episode_picked"] = jnp.logical_or(info["episode_picked"], picked)
    dropped = (box_pos[2] < 0.05) & info["episode_picked"]
    reward += dropped.astype(float) * -0.1
    out_of_bounds = jnp.any(jnp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    done = (
        out_of_bounds
        | jnp.isnan(mjx_data.qpos).any()
        | jnp.isnan(mjx_data.qvel).any()
        | dropped
    )
    info["_steps"] += oracle._config.action_repeat  # pylint: disable=protected-access
    info["_steps"] = jnp.where(
        done | (info["_steps"] >= oracle._config.episode_length),  # pylint: disable=protected-access
        0,
        info["_steps"],
    )

    metrics = dict(state.metrics)
    metrics.update(**rewards, out_of_bounds=out_of_bounds.astype(float))
    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info
    )
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(float),
        info=info,
        metrics=metrics,
    )


def _step_mujoco_aloha_peg_oracle(
    oracle: oracle_aloha_peg.SinglePegInsertion,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    delta = action * oracle._config.action_scale  # pylint: disable=protected-access
    ctrl = np.asarray(data.ctrl) + delta
    ctrl = np.clip(
        ctrl,
        np.asarray(oracle._lowers),  # pylint: disable=protected-access
        np.asarray(oracle._uppers),  # pylint: disable=protected-access
    )
    data.ctrl[:] = ctrl
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)
    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)

    socket_entrance_pos = mjx_data.site_xpos[oracle._socket_entrance_site]  # pylint: disable=protected-access
    socket_rear_pos = mjx_data.site_xpos[oracle._socket_rear_site]  # pylint: disable=protected-access
    peg_end2_pos = mjx_data.site_xpos[oracle._peg_end2_site]  # pylint: disable=protected-access
    socket_ab = socket_entrance_pos - socket_rear_pos
    socket_t = jnp.dot(peg_end2_pos - socket_rear_pos, socket_ab)
    socket_t /= jnp.dot(socket_ab, socket_ab) + 1e-6
    nearest_pt = socket_rear_pos + socket_t * socket_ab
    peg_end2_dist_to_line = jnp.linalg.norm(peg_end2_pos - nearest_pt)

    socket_pos = mjx_data.xpos[oracle._socket_body]  # pylint: disable=protected-access
    peg_pos = mjx_data.xpos[oracle._peg_body]  # pylint: disable=protected-access
    out_of_bounds = jnp.any(jnp.abs(socket_pos) > 1.0)
    out_of_bounds |= jnp.any(jnp.abs(peg_pos) > 1.0)
    raw_rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data, use_peg_insertion_reward=(peg_end2_dist_to_line < 0.005)
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in raw_rewards.items()
    }
    reward = sum(rewards.values()) / sum(
        oracle._config.reward_config.scales.values()  # pylint: disable=protected-access
    )
    done = (
        out_of_bounds
        | jnp.isnan(mjx_data.qpos).any()
        | jnp.isnan(mjx_data.qvel).any()
    )
    metrics = dict(state.metrics)
    metrics.update(
        **rewards,
        peg_end2_dist_to_line=peg_end2_dist_to_line,
        out_of_bounds=out_of_bounds.astype(float),
    )
    obs = oracle._get_obs(mjx_data)  # pylint: disable=protected-access
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(float),
        info=state.info,
        metrics=metrics,
    )


def _step_mujoco_panda_cartesian_oracle(
    oracle: oracle_panda_pick_cartesian.PandaPickCubeCartesian,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    info = dict(state.info)
    metrics = dict(state.metrics)
    action = np.asarray(action, dtype=np.float64)
    action_history = jnp.roll(info["action_history"], 1).at[0].set(action[2])
    info["action_history"] = action_history
    action = action.copy()
    action[2] = float(np.asarray(action_history[0]))

    newly_reset = info["_steps"] == 0
    info["newly_reset"] = newly_reset
    info["prev_reward"] = jnp.where(newly_reset, 0.0, info["prev_reward"])
    info["current_pos"] = jnp.where(
        newly_reset,
        oracle._start_tip_transform[:3, 3],  # pylint: disable=protected-access
        info["current_pos"],
    )
    info["reached_box"] = jnp.where(newly_reset, 0.0, info["reached_box"])
    info["prev_action"] = jnp.where(
        newly_reset, jnp.zeros(3), info["prev_action"]
    )

    increment = jnp.zeros(4)
    increment = increment.at[1:].set(jnp.asarray(action))
    ctrl, new_tip_position, no_soln = oracle._move_tip(  # pylint: disable=protected-access
        info["current_pos"],
        oracle._start_tip_transform[:3, :3],  # pylint: disable=protected-access
        jnp.asarray(data.ctrl),
        increment,
    )
    ctrl = np.clip(
        np.asarray(ctrl),
        np.asarray(oracle._lowers),  # pylint: disable=protected-access
        np.asarray(oracle._uppers),  # pylint: disable=protected-access
    )
    data.ctrl[:] = ctrl
    info["current_pos"] = new_tip_position

    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)
    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)

    raw_rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data, info
    )
    rewards = {
        key: value * oracle._config.reward_config.reward_scales[key]  # pylint: disable=protected-access
        for key, value in raw_rewards.items()
    }

    hand_box = (
        mjx_data.sensordata[
            oracle._mj_model.sensor_adr[  # pylint: disable=protected-access
                oracle._box_hand_found_sensor  # pylint: disable=protected-access
            ]
        ]
        > 0
    )
    raw_rewards["no_box_collision"] = jnp.where(hand_box, 0.0, 1.0)
    total_reward = jnp.clip(sum(rewards.values()), -1e4, 1e4)

    da = jnp.linalg.norm(jnp.asarray(action) - info["prev_action"])
    info["prev_action"] = jnp.asarray(action)
    total_reward += oracle._config.reward_config.action_rate * da  # pylint: disable=protected-access
    total_reward += no_soln * oracle._config.reward_config.no_soln_reward  # pylint: disable=protected-access

    box_pos = mjx_data.xpos[oracle._obj_body]  # pylint: disable=protected-access
    lifted = (box_pos[2] > 0.05) * oracle._config.reward_config.lifted_reward  # pylint: disable=protected-access
    total_reward += lifted
    success = oracle._get_success(  # pylint: disable=protected-access
        mjx_data, info
    )
    total_reward += success * oracle._config.reward_config.success_reward  # pylint: disable=protected-access

    reward = jnp.maximum(total_reward - info["prev_reward"], 0.0)
    info["prev_reward"] = jnp.maximum(total_reward, info["prev_reward"])
    reward = jnp.where(newly_reset, 0.0, reward)

    out_of_bounds = jnp.any(jnp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    metrics.update(out_of_bounds=out_of_bounds.astype(float))
    metrics.update({
        f"reward/{key}": value for key, value in raw_rewards.items()
    })
    metrics.update({
        "reward/lifted": lifted.astype(float),
        "reward/success": success.astype(float),
    })
    done = (
        out_of_bounds
        | jnp.isnan(mjx_data.qpos).any()
        | jnp.isnan(mjx_data.qvel).any()
        | success
    )
    info["_steps"] += oracle._config.action_repeat  # pylint: disable=protected-access
    info["_steps"] = jnp.where(
        done | (info["_steps"] >= oracle._config.episode_length),  # pylint: disable=protected-access
        0,
        info["_steps"],
    )
    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info
    )
    obs = jnp.concat([obs, no_soln.reshape(1), jnp.asarray(action)], axis=0)
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done.astype(float),
        info=info,
        metrics=metrics,
    )


def _step_mujoco_panda_pick_oracle(
    oracle: oracle_panda_pick.PandaPickCube,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    info = dict(state.info)
    metrics = dict(state.metrics)
    ctrl = data.ctrl + action * oracle._config.action_scale  # pylint: disable=protected-access
    data.ctrl[:] = np.clip(
        ctrl,
        np.asarray(oracle._lowers),  # pylint: disable=protected-access
        np.asarray(oracle._uppers),  # pylint: disable=protected-access
    )
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)

    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)
    raw_rewards = oracle._get_reward(  # pylint: disable=protected-access
        mjx_data, info
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in raw_rewards.items()
    }
    reward = jnp.clip(sum(rewards.values()), -1e4, 1e4)
    box_pos = mjx_data.xpos[oracle._obj_body]  # pylint: disable=protected-access
    out_of_bounds = jnp.any(jnp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    done = out_of_bounds | jnp.isnan(mjx_data.qpos).any()
    done |= jnp.isnan(mjx_data.qvel).any()
    done = done.astype(float)

    metrics.update(**raw_rewards, out_of_bounds=out_of_bounds.astype(float))
    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info
    )
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done,
        info=info,
        metrics=metrics,
    )


def _step_mujoco_panda_open_cabinet_oracle(
    oracle: oracle_panda_open_cabinet.PandaOpenCabinet,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    info = dict(state.info)
    metrics = dict(state.metrics)
    ctrl = data.ctrl + action * oracle._config.action_scale  # pylint: disable=protected-access
    data.ctrl[:] = np.clip(
        ctrl,
        np.asarray(oracle._lowers),  # pylint: disable=protected-access
        np.asarray(oracle._uppers),  # pylint: disable=protected-access
    )
    for _ in range(oracle.n_substeps):
        mujoco.mj_step(oracle.mj_model, data)

    mjx_data = _sync_mjx_data_from_mujoco(state.data, data, oracle.mj_model)
    raw_rewards = oracle._get_rewards(  # pylint: disable=protected-access
        mjx_data, info
    )
    rewards = {
        key: value * oracle._config.reward_config.scales[key]  # pylint: disable=protected-access
        for key, value in raw_rewards.items()
    }
    reward = jnp.clip(sum(rewards.values()), -1e4, 1e4)
    box_pos = mjx_data.xpos[oracle._obj_body]  # pylint: disable=protected-access
    out_of_bounds = jnp.any(jnp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    done = out_of_bounds | jnp.isnan(mjx_data.qpos).any()
    done |= jnp.isnan(mjx_data.qvel).any()
    done = done.astype(float)

    metrics.update(**raw_rewards, out_of_bounds=out_of_bounds.astype(float))
    obs = oracle._get_obs(  # pylint: disable=protected-access
        mjx_data, info
    )
    return state.replace(
        data=mjx_data,
        obs=obs,
        reward=reward,
        done=done,
        info=info,
        metrics=metrics,
    )


def _step_mujoco_oracle(
    oracle: Any,
    state: oracle_mjx_env.State,
    action: np.ndarray,
    data: mujoco.MjData,
) -> oracle_mjx_env.State:
    if isinstance(oracle, oracle_aloha_handover.HandOver):
        return _step_mujoco_aloha_handover_oracle(oracle, state, action, data)
    if isinstance(oracle, oracle_aloha_peg.SinglePegInsertion):
        return _step_mujoco_aloha_peg_oracle(oracle, state, action, data)
    if isinstance(oracle, oracle_panda_open_cabinet.PandaOpenCabinet):
        return _step_mujoco_panda_open_cabinet_oracle(
            oracle, state, action, data
        )
    if isinstance(oracle, oracle_panda_pick_cartesian.PandaPickCubeCartesian):
        return _step_mujoco_panda_cartesian_oracle(oracle, state, action, data)
    if isinstance(oracle, oracle_panda_pick.PandaPickCube):
        return _step_mujoco_panda_pick_oracle(oracle, state, action, data)
    if isinstance(
        oracle,
        (
            oracle_leap_rotate.CubeRotateZAxis,
            oracle_aero_rotate.CubeRotateZAxis,
        ),
    ):
        return _step_mujoco_hand_rotate_oracle(oracle, state, action, data)
    if isinstance(oracle, oracle_leap_reorient.CubeReorient):
        return _step_mujoco_leap_reorient_oracle(oracle, state, action, data)
    if isinstance(oracle, oracle_apollo.Joystick):
        return _step_mujoco_apollo_oracle(oracle, state, action, data)
    if isinstance(oracle, oracle_barkour.Joystick):
        return _step_mujoco_barkour_oracle(oracle, state, action, data)
    if isinstance(oracle, oracle_berkeley.Joystick):
        return _step_mujoco_berkeley_oracle(oracle, state, action, data)
    if isinstance(oracle, oracle_g1.Joystick):
        return _step_mujoco_g1_oracle(oracle, state, action, data)
    if isinstance(oracle, oracle_t1.Joystick):
        return _step_mujoco_t1_oracle(oracle, state, action, data)
    if isinstance(oracle, oracle_op3.Joystick):
        return _step_mujoco_op3_oracle(oracle, state, action, data)
    if isinstance(oracle, oracle_h1_inplace.InplaceGaitTracking):
        return _step_mujoco_h1_inplace_oracle(oracle, state, action, data)
    if isinstance(oracle, oracle_h1_joystick.JoystickGaitTracking):
        return _step_mujoco_h1_joystick_oracle(oracle, state, action, data)
    if isinstance(oracle, oracle_spot_joystick.Joystick):
        return _step_mujoco_spot_joystick_oracle(oracle, state, action, data)
    if isinstance(oracle, oracle_spot_getup.Getup):
        return _step_mujoco_spot_getup_oracle(oracle, state, action, data)
    if isinstance(oracle, oracle_spot_gait.JoystickGaitTracking):
        return _step_mujoco_spot_gait_oracle(oracle, state, action, data)
    if isinstance(oracle, oracle_go1_getup.Getup):
        return _step_mujoco_getup_oracle(oracle, state, action, data)
    if isinstance(oracle, oracle_go1_handstand.Handstand):
        return _step_mujoco_handstand_oracle(oracle, state, action, data)
    return _step_mujoco_joystick_oracle(oracle, state, action, data)


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


def _assert_obs_close(
    test: absltest.TestCase,
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
        atol=_STRICT_OBS_ATOL,
        rtol=_STRICT_OBS_RTOL,
        err_msg="obs[state]",
    )

    if not isinstance(actual, dict) or "privileged_state" not in actual:
        return
    actual_privileged = np.asarray(actual["privileged_state"][0])
    expected_privileged = np.asarray(expected["privileged_state"])
    np.testing.assert_allclose(
        actual_privileged,
        expected_privileged,
        atol=_STRICT_OBS_ATOL,
        rtol=_STRICT_OBS_RTOL,
        err_msg="obs[privileged_state]",
    )


def _mujoco_state_atol(task_id: str, key: str) -> float:
    if key == "qvel" and task_id == "LeapCubeRotateZAxis-v1" and _is_arm64():
        # The aarch64 libm/MuJoCo path leaves a 3.5e-12 residual in two hinge
        # velocities after the third step; qpos, ctrl, obs, reward, and info
        # stay at the strict 1e-12 bar.
        return _LEAP_ARM64_QVEL_ATOL
    return _STRICT_MUJOCO_STATE_ATOL


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
        atol = _mujoco_state_atol(task_id, key)
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
    max_mismatch_ratio: float = 0.025,
    max_ignored_abs_diff: int = 3,
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"frame shapes differ: {actual.shape} != {expected.shape}"
        )
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    if diff.size == 0:
        return
    mean_abs_diff = float(diff.mean())
    significant = diff > max_ignored_abs_diff
    mismatch_ratio = float(np.count_nonzero(significant)) / float(diff.size)
    if mean_abs_diff > max_mean_abs_diff:
        raise AssertionError(
            "mean render delta "
            f"{mean_abs_diff:.3f} exceeded {max_mean_abs_diff:.3f}; "
            f"max_delta={int(diff.max())}; "
            f"ignored_abs_diff={max_ignored_abs_diff}"
        )
    if mismatch_ratio > max_mismatch_ratio:
        raise AssertionError(
            "render mismatch ratio "
            f"{mismatch_ratio:.4%} exceeded {max_mismatch_ratio:.4%}; "
            f"mean_delta={mean_abs_diff:.3f}; max_delta={int(diff.max())}; "
            f"ignored_abs_diff={max_ignored_abs_diff}"
        )


def _official_render_kwargs(task_id: str) -> dict[str, Any]:
    if task_id == "ApolloJoystickFlatTerrain-v1":
        # Apollo's humanoid mesh has many thin STL edges at the low render-test
        # resolution. With the same fixed camera and bitwise-aligned state the
        # mean error stays below 1/255, but edge antialiasing leaves more >3/255
        # channels than the compact Go1 meshes.
        return {"max_mismatch_ratio": 0.05}
    if task_id == "Go1Getup-v1":
        # Getup uses the full-collision Go1 mesh set. The rendered frame is
        # state-aligned, but GL rasterization leaves a tiny number of edge
        # channels above the 3/255 rounding budget.
        return {"max_mismatch_ratio": 0.01}
    if task_id.startswith("G1"):
        # G1 has many small visual edges at the compact render-test resolution.
        # The raw mean delta remains under 1/255 after state sync; the wider
        # channel budget is only for >3/255 rasterization edge pixels.
        if (
            task_id == "G1JoystickRoughTerrain-v1"
            and platform.system() == "Windows"
        ):
            # Windows llvmpipe leaves slightly more textured hfield edge energy
            # while preserving state and low mean pixel error.
            return {"max_mean_abs_diff": 1.05, "max_mismatch_ratio": 0.095}
        return {"max_mismatch_ratio": 0.095}
    if task_id.startswith("H1"):
        # H1's STL edges leave a small low-resolution rasterization fringe after
        # the <=3/255 channel budget; the mean delta stays below 1/255.
        return {"max_mismatch_ratio": 0.07}
    if task_id == "SpotGetup-v1":
        # SpotGetup uses the full Boston Dynamics OBJ visual mesh. State and
        # camera are aligned, but the dense slanted mesh edges leave a visible
        # low-resolution antialiasing fringe between EnvPool's source-built
        # renderer and the official MuJoCo wheel.
        return {"max_mean_abs_diff": 4.2, "max_mismatch_ratio": 0.10}
    if task_id == "PandaPickCubeCartesian-v1":
        # The Cartesian camera scene has high-contrast gripper/cube edges close
        # to the fixed camera. State and mean pixel error are aligned, while a
        # small edge fringe crosses the >3/255 channel threshold.
        return {"max_mismatch_ratio": 0.035}
    if task_id == "Op3Joystick-v1" and platform.system() == "Windows":
        # OP3's filesystem-loaded visual meshes align with the official model;
        # Windows llvmpipe leaves a narrow extra fringe above the <=3/255 budget.
        return {"max_mismatch_ratio": 0.03}
    if "RoughTerrain" in task_id:
        # Rough terrain renders a textured hfield. EnvPool's offscreen renderer
        # and Python `mujoco.Renderer` agree on low mean pixel error. After
        # ignoring <= 3/255 GL rounding, the remaining differences are mostly
        # textured floor and robot-edge channels at low test resolution.
        return {"max_mismatch_ratio": 0.055}
    return {}


def _uses_envpool_native_physics_oracle(task_id: str) -> bool:
    # These tasks have a known upstream-backend boundary where testing the
    # official formulas on EnvPool's native MuJoCo state is the meaningful
    # correctness check. Panda Cartesian's boundary is the analytical IK action
    # projection: upstream JAX emits float32 joint targets, while EnvPool runs a
    # C++ port before stepping MuJoCo, leaving only ~4e-8 joint-state residuals.
    return task_id in (
        "BerkeleyHumanoidJoystickRoughTerrain-v1",
        "G1JoystickRoughTerrain-v1",
        "PandaPickCubeCartesian-v1",
        "PandaRobotiqPushCube-v1",
    )


_OP3_RENDER_MODEL: mujoco.MjModel | None = None


def _official_render_model(task_id: str, oracle: Any) -> mujoco.MjModel:
    if task_id != "Op3Joystick-v1":
        return oracle.mj_model
    global _OP3_RENDER_MODEL
    if _OP3_RENDER_MODEL is None:
        # The upstream OP3 Python oracle loads XML from an in-memory asset dict.
        # MuJoCo's asset dict keys collide for OP3 because visual meshes and
        # simplified collision meshes share basenames, so that path renders the
        # simplified collision body as the visual body. For render correctness,
        # load the same pinned XML/assets from filesystem runfiles instead.
        xml_path = _find_runfile(
            "envpool/mujoco/playground/assets/mujoco_playground/_src/"
            "locomotion/op3/xmls/scene_mjx_feetonly.xml"
        )
        _OP3_RENDER_MODEL = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    return _OP3_RENDER_MODEL


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
    def __init__(self, oracle_or_model: Any, task_id: str):
        model = (
            oracle_or_model
            if isinstance(oracle_or_model, mujoco.MjModel)
            else oracle_or_model.mj_model
        )
        self._camera = _official_render_camera(task_id)
        self._renderer = mujoco.Renderer(
            model, height=_RENDER_HEIGHT, width=_RENDER_WIDTH
        )

    def render(self, data: mujoco.MjData) -> np.ndarray:
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
    _assert_obs_close(test, obs, state.obs)
    # EnvPool exposes `reward` as float32 by API contract, so compare the
    # official scalar after the same public dtype conversion instead of allowing
    # a wider numeric tolerance.
    reward_report = {
        key: (
            float(np.asarray(info[key][0])),
            float(np.asarray(state.metrics[_oracle_metric_name(key, task_id)])),
        )
        for key in _reward_info_keys(task_id)
    }
    np.testing.assert_array_equal(
        reward[0],
        np.asarray(state.reward, dtype=reward.dtype),
        err_msg=f"reward components={reward_report}",
    )
    test.assertEqual(bool(terminated[0]), bool(state.done))
    test.assertFalse(bool(truncated[0]))
    if "command" in info:
        np.testing.assert_array_equal(
            info["command"][0],
            np.asarray(state.info["command"]),
        )
    if "steps_until_next_cmd" in info:
        test.assertEqual(
            int(info["steps_until_next_cmd"][0]),
            int(np.asarray(state.info["steps_until_next_cmd"])),
        )
    test.assertEqual(bool(info["terminated"][0]), bool(np.asarray(state.done)))
    for key in _reward_info_keys(task_id):
        np.testing.assert_allclose(
            info[key][0],
            np.asarray(state.metrics[_oracle_metric_name(key, task_id)]),
            atol=5e-11,
            rtol=5e-11,
            err_msg=f"info[{key}]",
        )


def _compare_berkeley_pip_and_native_one_step(task_id: str) -> dict[str, float]:
    oracle = _make_oracle(task_id)
    env = make_gymnasium(task_id, num_envs=1, seed=0, **_env_config(task_id))
    try:
        _, info = env.reset()
        data = mujoco.MjData(oracle.mj_model)
        data.qpos[:] = info["qpos"][0, : oracle.mj_model.nq]
        data.qvel[:] = info["qvel"][0, : oracle.mj_model.nv]
        data.ctrl[:] = info["ctrl"][0, : oracle.mj_model.nu]
        mujoco.mj_forward(oracle.mj_model, data)
        reset_qacc = float(
            np.max(np.abs(data.qacc - info["qacc"][0, : oracle.mj_model.nv]))
        )
        reset_sensordata = float(
            np.max(
                np.abs(
                    data.sensordata
                    - info["sensordata"][0, : data.sensordata.size]
                )
            )
        )
        action = _make_actions(1, oracle.action_size)[0]
        data.ctrl[:] = (
            np.asarray(oracle._default_pose)  # pylint: disable=protected-access
            + action[0] * oracle._config.action_scale  # pylint: disable=protected-access
        )
        for _ in range(oracle.n_substeps):
            mujoco.mj_step(oracle.mj_model, data)
        _, _, _, _, info = env.step(action)
        return {
            "reset_qacc": reset_qacc,
            "reset_sensordata": reset_sensordata,
            "qpos": float(
                np.max(
                    np.abs(info["qpos"][0, : oracle.mj_model.nq] - data.qpos)
                )
            ),
            "qvel": float(
                np.max(
                    np.abs(info["qvel"][0, : oracle.mj_model.nv] - data.qvel)
                )
            ),
        }
    finally:
        env.close()


def _assert_berkeley_rough_hfield_backend_gap(
    test: absltest.TestCase, rough: dict[str, float]
) -> None:
    reset_exact = (
        rough["reset_qacc"] <= 5e-12 and rough["reset_sensordata"] <= 5e-12
    )
    small_backend_gap = rough["qpos"] <= 2e-5 and rough["qvel"] <= 5e-4
    large_backend_gap = (
        1e-4 < rough["qpos"] <= 2e-3 and 1e-3 < rough["qvel"] <= 7e-2
    )
    if reset_exact and (small_backend_gap or large_backend_gap):
        return

    if _is_arm64():
        # The aarch64 hfield contact path can diverge from the PyPI wheel
        # already at reset forward(), while flat terrain remains exact below.
        # Keep this as a bounded backend diagnostic; the actual task alignment
        # test validates EnvPool state with the official obs/reward formulas.
        for key, bound in (
            ("reset_qacc", 2.0),
            ("reset_sensordata", 5e-2),
            ("qpos", 2e-3),
            ("qvel", 7e-2),
        ):
            test.assertLessEqual(rough[key], bound, rough)
        return

    test.fail(f"unexpected Berkeley rough hfield backend gap: {rough}")


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

    def test_berkeley_hfield_backend_gap_is_not_formula_drift(self) -> None:
        """Pins Berkeley rough terrain to a bounded hfield backend gap."""
        if not _is_primary_shard():
            return
        flat = _compare_berkeley_pip_and_native_one_step(
            "BerkeleyHumanoidJoystickFlatTerrain-v1"
        )
        rough = _compare_berkeley_pip_and_native_one_step(
            "BerkeleyHumanoidJoystickRoughTerrain-v1"
        )
        for key in ("reset_qacc", "reset_sensordata"):
            self.assertLessEqual(flat[key], 5e-12, flat)
        for key in ("qpos", "qvel"):
            self.assertLessEqual(flat[key], 1e-12, flat)
        _assert_berkeley_rough_hfield_backend_gap(self, rough)

    def test_rollout_and_render_align_with_oracle(self) -> None:
        """Checks reset and three rendered control steps against the oracle."""
        for task_id in _TASK_IDS:
            with self.subTest(task_id=task_id):
                oracle = _make_oracle(task_id)
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
                    _assert_obs_close(self, obs, state.obs)
                    if task_id == "Op3Joystick-v1":
                        render_model = _official_render_model(task_id, oracle)
                        render_data = _make_mujoco_data_from_model(
                            render_model, info
                        )
                    else:
                        render_model = oracle
                        render_data = mujoco_data
                    official_renderer = _OfficialFrameRenderer(
                        render_model, task_id
                    )
                    render_kwargs = _official_render_kwargs(task_id)
                    rendered = env.render(env_ids=[0])
                    assert rendered is not None
                    _assert_frames_close(
                        rendered[0],
                        official_renderer.render(render_data),
                        **render_kwargs,
                    )

                    for step, action in enumerate(actions, start=1):
                        if _uses_envpool_native_physics_oracle(task_id):
                            obs, reward, terminated, truncated, info = env.step(
                                action
                            )
                            state = _step_oracle_from_envpool_info(
                                oracle, state, action[0], info
                            )
                            mujoco_data = _make_mujoco_data_from_info(
                                oracle, info
                            )
                        else:
                            state = _step_mujoco_oracle(
                                oracle, state, action[0], mujoco_data
                            )
                            obs, reward, terminated, truncated, info = env.step(
                                action
                            )
                        if task_id == "Op3Joystick-v1":
                            render_data = _make_mujoco_data_from_model(
                                render_model, info
                            )
                        else:
                            render_data = mujoco_data
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
                                official_renderer.render(render_data),
                                **render_kwargs,
                            )
                finally:
                    if official_renderer is not None:
                        official_renderer.close()
                    env.close()


if __name__ == "__main__":
    absltest.main()
