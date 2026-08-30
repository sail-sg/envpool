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
"""Isolated, pinned Python 3.12 oracle. Never imported by runtime code."""

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

# Configure GL before importing either MuJoCo or MJLab. Physics remains the
# unmodified official CPU MuJoCo-Warp implementation, including its JIT.
os.environ["WANDB_MODE"] = "disabled"
if platform.system() == "Linux":
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")

from envpool.mujoco.oracle import configure_mujoco_package_shared_lib
from envpool.mujoco.render_oracle import (
    configure_macos_mujoco_renderer,
    configure_windows_mujoco_renderer,
)
from envpool.python.glfw_context import preload_windows_gl_dlls
from third_party.mjlab.oracle_util import configure_cache


def synchronize(env: Any, source: Any, folder: Path) -> Any:
    """One reset-time synchronization of model, physics, managers and RNG.

    No step in the rollout calls this function. Native randomization is tested
    separately without using this function or an oracle state overwrite.
    """
    import mujoco
    import numpy as np
    import torch
    import warp as wp

    state = json.loads(str(source["state"]))
    env.common_step_counter = state["total_steps"]
    for key in source.files:
        if not key.startswith("physics:"):
            continue
        name = key.removeprefix("physics:")
        kind, *parts = name.split(".")
        target = env.sim.wp_model if kind == "model" else env.sim.wp_data
        for part in parts:
            target = getattr(target, part)
        array = target.numpy()
        value = np.frombuffer(source[key], dtype=array.dtype).reshape(
            array.shape
        )
        wp.copy(target, wp.array(value, dtype=target.dtype, device="cpu"))

    # Ray casting owns a mesh/BVH representation in addition to the physics
    # hfield arrays. Rebuild it once from the same randomized reset model.
    model_path = folder / "reset.mjb"
    model_path.write_bytes(source["model"].tobytes())
    native_model = mujoco.MjModel.from_binary_path(str(model_path))
    for name in dir(native_model):
        value = getattr(native_model, name)
        if isinstance(value, np.ndarray):
            target = getattr(env.sim.mj_model, name)
            if target.flags.writeable:
                np.copyto(target, value)
            else:
                np.testing.assert_array_equal(target, value, err_msg=name)
    context = env.sim._sensor_context
    if context is not None:
        context.recreate(env.sim.mj_model, env.sim.expanded_fields)

    for name, values in state["entities"].items():
        env.scene[name].data.encoder_bias[:] = torch.tensor(
            values["encoder_bias"]
        )
    env.scene.env_origins[:] = torch.tensor(state["origin"])
    if env.cfg.commands:
        command = env.command_manager.get_term(next(iter(env.cfg.commands)))
        command.time_left[:] = state["command"]["time_left"]
        command.command_counter[:] = state["command"]["counter"]
        for name in (
            "target_pos",
            "episode_success",
            "target_selection",
            "vel_command_b",
            "vel_command_w",
            "heading_target",
            "heading_error",
            "is_heading_env",
            "is_standing_env",
            "is_world_env",
            "is_forward_env",
            "time_steps",
            "body_pos_relative_w",
            "body_quat_relative_w",
            "bin_failed_count",
        ):
            if name in state["command"] and hasattr(command, name):
                target = getattr(command, name)
                target[:] = torch.tensor(
                    state["command"][name], dtype=target.dtype
                ).reshape(target.shape)
        if hasattr(command, "_pending_forward"):
            command._pending_forward = False
        if hasattr(command, "_current_bin_failed"):
            command._current_bin_failed[:] = torch.tensor(
                state["command"]["current_bin_failed"]
            )
        if "peak_heights" in state["command"]:
            term = env.reward_manager.get_term_cfg("foot_swing_height").func
            term.peak_heights[:] = torch.tensor(
                state["command"]["peak_heights"]
            ).reshape(1, -1)
        if "terrain" in state["command"] and hasattr(
            env.scene.terrain, "terrain_levels"
        ):
            terrain = env.scene.terrain
            terrain.terrain_levels[:] = state["command"]["terrain"]["level"]
            terrain.terrain_types[:] = state["command"]["terrain"]["type"]
    for name, timer in state["event_timers"].items():
        index = env.event_manager._mode_term_names["interval"].index(name)
        env.event_manager._interval_term_time_left[index][:] = timer
    env.sim.forward()
    if env.cfg.commands:
        command._update_metrics()
        if hasattr(command, "_cached_target_obj_pos"):
            command._cached_target_obj_pos[:] = torch.tensor(
                state["command"]["cached_target_obj_pos"]
            ).reshape(1, 3)
    for name, values in state["contacts"].items():
        sensor = env.scene[name]
        if sensor._air_time_state is not None:
            for key in (
                "current_air_time",
                "last_air_time",
                "current_contact_time",
                "last_contact_time",
            ):
                target = getattr(sensor._air_time_state, key)
                target[:] = torch.tensor(values[key]).reshape(target.shape)
        if sensor._history_state is not None:
            for key, target in sensor._history_state.items():
                target[:] = torch.tensor(values[key + "_history"]).reshape(
                    target.shape
                )
        sensor._cache_valid = False
    env.sim.sense()
    noise_count = sum(
        np.prod(env.observation_manager._group_obs_term_dim[group][i])
        for group, terms in env.observation_manager._group_obs_term_cfgs.items()
        for i, term in enumerate(terms)
        if term.noise is not None
    )
    torch.manual_seed(state["seed"])
    torch.rand(int(state["rng_draws"] - noise_count))
    return env.observation_manager.compute(update_history=True)


def main() -> None:
    """Query the registry or replay a native reset using official MJLab."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--registry", action="store_true")
    args = parser.parse_args()
    configure_cache(args.cache)
    preload_windows_gl_dlls(strict=True)
    # Linux wheels bind MjSpec's C++ containers with libc++, whereas the native
    # engine can use libstdc++. Keep the wheel's engine there, as other MuJoCo
    # family oracles do; swapping those libraries invalidates the MjSpec ABI.
    configure_mujoco_package_shared_lib()
    configure_macos_mujoco_renderer()
    configure_windows_mujoco_renderer()

    import numpy as np
    import torch
    import warp as wp

    wp.config.kernel_cache_dir = str(args.cache)
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import list_tasks, load_env_cfg
    from mjlab.viewer.offscreen_renderer import OffscreenRenderer

    if args.registry:
        args.output.write_text(json.dumps(list_tasks()))
        return
    with np.load(args.input, allow_pickle=False) as source:
        task = str(source["task"])
        cfg = load_env_cfg(task)
        cfg.scene.num_envs = 1
        cfg.seed = 17
        cfg.auto_reset = False
        cfg.viewer.width, cfg.viewer.height = map(
            int, source.get("render_size", (96, 80))
        )
        if "motion" in cfg.commands:
            cfg.commands["motion"].motion_file = str(source["motion_file"])
        env = ManagerBasedRlEnv(cfg, device="cpu")
        renderer: Any = None
        try:
            env.reset()
            obs = synchronize(env, source, args.output.parent)
            if bool(source.get("reset_after_sync", False)):
                obs, _ = env.reset()
            rows: dict[str, list] = {}
            frames, frame_steps = [], []
            render_steps = set(source["render_steps"].tolist())

            def record(
                step: int, reward: float, terminated: bool, truncated: bool
            ) -> None:
                values = {f"obs:{k}": v.numpy().copy() for k, v in obs.items()}
                values.update(
                    reward=np.array(reward, np.float32),
                    terminated=np.array(terminated),
                    truncated=np.array(truncated),
                    elapsed_step=env.episode_length_buf.numpy().copy(),
                    qpos=env.sim.data.qpos.numpy().copy(),
                    qvel=env.sim.data.qvel.numpy().copy(),
                )
                for key, value in values.items():
                    rows.setdefault(key, []).append(value)
                if step in render_steps:
                    nonlocal renderer
                    if renderer is None:
                        renderer = OffscreenRenderer(
                            env.sim.mj_model,
                            cfg.viewer,
                            env.scene,
                            env.sim.model,
                            env.sim.expanded_fields,
                        )
                        renderer.initialize()
                    renderer.update(env.sim.data)
                    # mjv_updateScene centers infinite planes using the previous
                    # GL camera. A second scene update settles that derived
                    # geometry without advancing or synchronizing physics.
                    renderer.update(env.sim.data)
                    frame = renderer.render()
                    if step == 0 and platform.system() == "Darwin":
                        for _ in range(4):
                            renderer.update(env.sim.data)
                            frame = renderer.render()
                    frames.append(frame.copy())
                    frame_steps.append(step)

            record(0, 0, False, False)
            for step, action in enumerate(source["actions"], 1):
                obs, reward, terminated, truncated, _ = env.step(
                    torch.from_numpy(action[None])
                )
                record(step, reward.item(), terminated.item(), truncated.item())
                if terminated.item() or truncated.item():
                    break
            result = {key: np.stack(value) for key, value in rows.items()}
            result.update(
                frames=np.stack(frames),
                frame_steps=np.array(frame_steps),
                action_shape=np.array(env.single_action_space.shape),
                action_low=env.single_action_space.low,
                action_high=env.single_action_space.high,
            )
            np.savez(args.output, **result)
        finally:
            if renderer is not None:
                renderer.close()
            env.close()


if __name__ == "__main__":
    main()
