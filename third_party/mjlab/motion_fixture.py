# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Local synthetic motion for tests, using the official CSV conversion math.

This is not a bundled default task motion or a published motion dataset. Never
call upstream run_sim/main here: those functions upload to a WandB registry.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.sim import Simulation


def generate_motion(cfg: Any, output: Path) -> None:
    """Convert a local synthetic G1 motion using upstream's reference math."""
    scene = Scene(cfg.scene, device="cpu")
    sim = Simulation(1, cfg.sim, spec=scene.spec, device="cpu")
    scene.initialize(sim.mj_model, sim.model, sim.data)
    robot = scene["robot"]
    frames = 401
    phase = np.arange(frames, dtype=np.float32) * (2 * np.pi / 200)
    root = robot.data.default_root_state[0].numpy()
    joints = robot.data.default_joint_pos[0].numpy()
    positions = np.broadcast_to(root[:3], (frames, 3)).copy()
    positions[:, 0] += 0.015 * np.sin(phase)
    positions[:, 2] += 0.002 * np.sin(phase * 2)
    quaternions = np.zeros((frames, 4), dtype=np.float32)
    # The official CSV contract uses xyzw; MotionLoader converts to wxyz.
    yaw = 0.015 * np.sin(phase)
    quaternions[:, 2] = np.sin(yaw / 2)
    quaternions[:, 3] = np.cos(yaw / 2)
    joint_positions = np.broadcast_to(joints, (frames, len(joints))).copy()
    for index, name in enumerate(robot.joint_names):
        if "shoulder_pitch" in name or "elbow" in name:
            joint_positions[:, index] += 0.025 * np.sin(phase + index * 0.1)
    csv = output.with_suffix(".csv")
    np.savetxt(
        csv,
        np.concatenate([positions, quaternions, joint_positions], axis=1),
        delimiter=",",
    )
    motion = MotionLoader(str(csv), input_fps=50, output_fps=50, device="cpu")
    values: dict[str, list] = {
        key: []
        for key in (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
    }
    for _ in range(motion.output_frames):
        (pos, quat, lin_vel, ang_vel, joint_pos, joint_vel), _ = (
            motion.get_next_state()
        )
        robot.write_root_state_to_sim(
            torch.cat([pos, quat, lin_vel, ang_vel], dim=-1)
        )
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.forward()
        scene.update(cfg.sim.mujoco.timestep)
        for key, value in (
            ("joint_pos", robot.data.joint_pos),
            ("joint_vel", robot.data.joint_vel),
            ("body_pos_w", robot.data.body_link_pos_w),
            ("body_quat_w", robot.data.body_link_quat_w),
            ("body_lin_vel_w", robot.data.body_link_lin_vel_w),
            ("body_ang_vel_w", robot.data.body_link_ang_vel_w),
        ):
            values[key].append(value[0].numpy().copy())
    np.savez_compressed(
        output,
        fps=np.array([50]),
        **{k: np.stack(v) for k, v in values.items()},
    )
