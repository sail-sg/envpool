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
"""Pinned Playground task construction for tests and doc rendering."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from unittest import mock

import mujoco
import numpy as np
from etils import epath
from mujoco_playground._src import mjx_env as oracle_mjx_env
from mujoco_playground._src.locomotion.apollo import joystick as oracle_apollo
from mujoco_playground._src.locomotion.barkour import joystick as oracle_barkour
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

from envpool.mujoco.oracle import runfiles_repository


class FrameRenderer:
    """Render official state with the same camera setup in tests and docs."""

    def __init__(self, oracle: Any, task_id: str, *, width: int, height: int):
        """Create an offscreen buffer matching the requested image size."""
        model = oracle.mj_model
        # Match native buffer dimensions: skybox projection on CGL depends on
        # the full offscreen buffer, not just the requested output viewport.
        model.vis.global_.offwidth = width
        model.vis.global_.offheight = height
        self._camera = _official_render_camera(task_id)
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, data: mujoco.MjData) -> np.ndarray:
        """Draw the current state after updating camera-dependent geometry."""
        # MuJoCo builds infinite-plane geometry before updating the camera.
        # Refresh that geometry at the selected camera before the single draw.
        self._renderer.update_scene(data, camera=self._camera)
        self._renderer.update_scene(data, camera=self._camera)
        return self._renderer.render()

    def close(self) -> None:
        """Release the offscreen renderer."""
        self._renderer.close()


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


def native_config(task_id: str) -> dict[str, Any]:
    """Native settings shared by oracle checks and documentation images."""

    def zero_command_kwargs(
        *,
        noise_key: str | None = "noise_level",
        noise_value: float = 0.0,
        disable_key: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "lin_vel_x_min": 0.0,
            "lin_vel_x_max": 0.0,
            "lin_vel_y_min": 0.0,
            "lin_vel_y_max": 0.0,
            "ang_vel_yaw_min": 0.0,
            "ang_vel_yaw_max": 0.0,
        }
        if noise_key is not None:
            kwargs[noise_key] = noise_value
        if disable_key is not None:
            kwargs[disable_key] = 0.0
        return kwargs

    if task_id.startswith("Apollo"):
        return {
            "noise_level": 0.0,
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
    if task_id.startswith("Barkour"):
        return {
            **zero_command_kwargs(noise_key="obs_noise", noise_value=-1.0),
            "kick_wait_steps_min": 100000,
            "kick_wait_steps_max": 100001,
        }
    if task_id.startswith("Op3"):
        return zero_command_kwargs(noise_key="obs_noise", noise_value=-1.0)
    if task_id.startswith(("BerkeleyHumanoid", "G1", "T1")):
        return zero_command_kwargs(disable_key="push_enable")
    if task_id.startswith("H1"):
        return zero_command_kwargs(noise_key="obs_noise_level")
    if task_id == "SpotFlatTerrainJoystick-v1":
        return zero_command_kwargs(disable_key="pert_enable")
    if task_id == "SpotGetup-v1":
        return {"noise_level": 0.0}
    if task_id == "SpotJoystickGaitTracking-v1":
        return zero_command_kwargs()
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
    return {}


def make_oracle(task_id: str) -> Any:
    """Construct the pinned official task with comparison noise disabled."""
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
        xml = find_runfile(
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
        module = oracle_berkeley
        return _make_humanoid_joystick(module, task_id)
    if task_id in (
        "G1JoystickFlatTerrain-v1",
        "G1JoystickRoughTerrain-v1",
    ):
        module = oracle_g1
        return _make_humanoid_joystick(module, task_id)
    if task_id in (
        "T1JoystickFlatTerrain-v1",
        "T1JoystickRoughTerrain-v1",
    ):
        module = oracle_t1
        return _make_humanoid_joystick(module, task_id)
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


def _make_humanoid_joystick(module: Any, task_id: str) -> Any:
    config = module.default_config()
    config.impl = "jax"
    config.noise_config.level = 0.0
    config.push_config.enable = False
    config.lin_vel_x = [0.0, 0.0]
    config.lin_vel_y = [0.0, 0.0]
    config.ang_vel_yaw = [0.0, 0.0]
    task = "rough_terrain" if "RoughTerrain" in task_id else "flat_terrain"
    return module.Joystick(task=task, config=config)


def _configure_oracle_assets() -> None:
    oracle_mjx_env.MENAGERIE_PATH = epath.Path(
        runfiles_repository("mujoco_menagerie_playground").as_posix()
    )


def find_runfile(relative: str) -> Path:
    """Resolve a Playground asset in directory or manifest-only runfiles."""
    manifest_path = _find_manifest_runfile(relative)
    if manifest_path is not None:
        return manifest_path
    roots = _candidate_runfile_roots()
    for root in roots:
        path = root / relative
        if path.exists():
            return path
    for root in roots:
        if root.is_dir():
            for workspace in root.iterdir():
                candidate = workspace / relative
                if candidate.exists():
                    return candidate
    raise FileNotFoundError(f"Could not find runfile {relative!r}")


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


def _candidate_runfile_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("TEST_SRCDIR", "RUNFILES_DIR", "PYTHON_RUNFILES"):
        if os.environ.get(env_name):
            roots.append(Path(os.environ[env_name]))
    roots.append(Path.cwd())
    roots.extend(Path(__file__).parents)
    roots.extend(Path(__file__).resolve().parents)
    workspace = os.environ.get("TEST_WORKSPACE")
    if workspace:
        roots.extend(root / workspace for root in list(roots))
    return list(dict.fromkeys(roots))
