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
    """One reset-time synchronization of model, physics and managers.

    No step in the rollout calls this function. Native randomization is tested
    separately without using this function or an oracle state overwrite.
    """
    import mujoco
    import numpy as np
    import torch
    import warp as wp

    state = json.loads(str(source["state"]))
    env.common_step_counter = state["total_steps"]
    env.curriculum_manager.compute(env_ids=torch.tensor([0], device="cpu"))
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
    return env.observation_manager.compute(update_history=True)


def native_sample_checks(env: Any, source: Any) -> tuple[Any, Any]:
    """Condition official stochastic events on checked native semantic state.

    Event scheduling, deterministic command updates, physics, observations and
    rewards still execute in the oracle. Only a triggered event may copy its
    selected target or reset pose; no RNG state or future output is replayed.
    """
    from unittest.mock import patch

    import numpy as np
    import torch
    from mjlab.utils.lab_api.math import (
        euler_xyz_from_quat,
        quat_apply,
        quat_inv,
        quat_mul,
    )

    states = [json.loads(str(value)) for value in source["step_states"]]
    metadata = json.loads(str(source["metadata"]))
    integrator = {
        key: source["native_" + key]
        for key in ("qpos", "qvel", "warmstart", "ctrl")
    }
    step = 0
    advance = env.sim.step

    def advance_from_checked_controls() -> None:
        # Independently validate the action mapping, then use the actual native
        # motor targets as physics inputs. Float32 FMA differences as small as
        # 6e-8 can change a contact solver's stopping point; this avoids widening
        # state/force budgets or reproducing Torch's arithmetic in native code.
        actual = integrator["ctrl"][step].reshape(env.sim.data.ctrl.shape)
        np.testing.assert_allclose(
            actual,
            env.sim.data.ctrl.numpy(),
            rtol=1e-6,
            atol=1e-6,
            err_msg="processed actuator targets",
        )
        env.sim.data.ctrl[:] = torch.as_tensor(actual)
        advance()

    env.sim.step = advance_from_checked_controls

    def within(value: Any, bounds: Any, label: str) -> None:
        value, bounds = np.asarray(value), np.asarray(bounds)
        assert np.isfinite(value).all(), label
        assert np.all(value >= bounds[..., 0] - 1e-4), label
        assert np.all(value <= bounds[..., 1] + 1e-4), label

    def pose(entity: str) -> tuple[Any, Any, Any, Any]:
        index = metadata["entities"][entity]["indexing"]
        q = integrator["qpos"][step, 0]
        v = integrator["qvel"][step, 0]
        qi = int(np.asarray(index["free_joint_q_adr"]).ravel()[0])
        vi = int(np.asarray(index["free_joint_v_adr"]).ravel()[0])
        position = torch.from_numpy(q[qi : qi + 3].copy())[None]
        rotation = torch.from_numpy(q[qi + 3 : qi + 7].copy())[None]
        linear = torch.from_numpy(v[vi : vi + 3].copy())[None]
        angular = quat_apply(
            rotation, torch.from_numpy(v[vi + 3 : vi + 6].copy())[None]
        )
        np.testing.assert_allclose(
            torch.linalg.vector_norm(rotation, dim=-1), 1, atol=1e-5
        )
        return position, rotation, linear, angular

    command = (
        env.command_manager.get_term(next(iter(env.cfg.commands)))
        if env.cfg.commands
        else None
    )
    if command is not None:
        original = command._resample_command

        def resample(env_ids: Any) -> None:
            current = states[step]["command"]
            previous = states[step - 1]["command"]
            wrapped = (
                hasattr(command, "motion")
                and command.time_steps.item() >= command.motion.time_step_total
            )
            if wrapped:
                np.testing.assert_allclose(
                    command.time_left.item(), current["time_left"], atol=1e-5
                )
            else:
                assert previous["time_left"] <= env.step_dt + 1e-5, (
                    "premature command resample"
                )
                within(
                    current["time_left"],
                    command.cfg.resampling_time_range,
                    "command duration",
                )

            probabilities = []
            multinomial = torch.multinomial

            def sample(weights: Any, *args: Any, **kwargs: Any) -> Any:
                probabilities.append(weights.clone())
                return multinomial(weights, *args, **kwargs)

            with patch.object(torch, "multinomial", sample):
                original(env_ids)
            command.time_left[:] = current["time_left"]

            if hasattr(command, "motion"):
                frame = current["time_steps"] - (0 if wrapped else 1)
                assert 0 <= frame < command.motion.time_step_total
                np.testing.assert_allclose(
                    probabilities[0].numpy(),
                    current["sampling_probabilities"],
                    rtol=1e-5,
                    atol=1e-6,
                )
                command.time_steps[:] = frame
                position, rotation, linear, angular = pose("robot")
                cfg = command.cfg
                xyz = ("x", "y", "z")
                angles = ("roll", "pitch", "yaw")
                within(
                    position - command.body_pos_w[:, 0],
                    [cfg.pose_range.get(k, (0, 0)) for k in xyz],
                    "motion position",
                )
                delta = quat_mul(rotation, quat_inv(command.body_quat_w[:, 0]))
                euler = torch.stack(euler_xyz_from_quat(delta), dim=-1)
                euler = (euler + np.pi) % (2 * np.pi) - np.pi
                within(
                    euler,
                    [cfg.pose_range.get(k, (0, 0)) for k in angles],
                    "motion orientation",
                )
                within(
                    linear - command.body_lin_vel_w[:, 0],
                    [cfg.velocity_range.get(k, (0, 0)) for k in xyz],
                    "motion linear velocity",
                )
                within(
                    angular - command.body_ang_vel_w[:, 0],
                    [cfg.velocity_range.get(k, (0, 0)) for k in angles],
                    "motion angular velocity",
                )
                index = metadata["entities"]["robot"]["indexing"]
                qids = np.asarray(index["joint_q_adr"], dtype=int).ravel()
                vids = np.asarray(index["joint_v_adr"], dtype=int).ravel()
                joints = torch.from_numpy(
                    integrator["qpos"][step, :, qids].copy()
                ).reshape(1, -1)
                velocities = torch.from_numpy(
                    integrator["qvel"][step, :, vids].copy()
                ).reshape(1, -1)
                limits = command.robot.data.soft_joint_pos_limits
                low = torch.clip(
                    command.joint_pos + cfg.joint_position_range[0],
                    limits[:, :, 0],
                    limits[:, :, 1],
                )
                high = torch.clip(
                    command.joint_pos + cfg.joint_position_range[1],
                    limits[:, :, 0],
                    limits[:, :, 1],
                )
                within(
                    joints, torch.stack((low, high), dim=-1), "motion joints"
                )
                np.testing.assert_allclose(
                    velocities, command.joint_vel, rtol=1e-5, atol=1e-5
                )
                command._write_reference_state_to_sim(
                    env_ids,
                    position,
                    rotation,
                    linear,
                    angular,
                    joints,
                    velocities,
                )
            elif hasattr(command, "vel_command_b"):
                if not current["is_standing_env"]:
                    within(
                        current["vel_command_w"],
                        [
                            getattr(command.cfg.ranges, k)
                            for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
                        ],
                        "velocity target",
                    )
                within(
                    current["heading_target"],
                    command.cfg.ranges.heading,
                    "heading target",
                )
                for name in (
                    "vel_command_w",
                    "heading_target",
                    "is_heading_env",
                    "is_standing_env",
                    "is_world_env",
                    "is_forward_env",
                ):
                    target = getattr(command, name)
                    target[:] = torch.as_tensor(
                        current[name], dtype=target.dtype
                    ).reshape(target.shape)
                # The world vector is the sampled input. Do not copy the
                # native derived body command: upstream still computes its
                # heading/rotation/standing updates from this starting value.
                command.vel_command_b[:] = command.vel_command_w
                if current["is_forward_env"]:
                    command.vel_command_b[:, 0] = (
                        command.vel_command_b[:, 0].abs().clamp(min=0.3)
                    )
                    command.vel_command_b[:, 1:] = 0
            else:
                cfg = command.cfg
                within(
                    np.asarray(current["target_pos"])
                    - np.asarray(states[step]["origin"]),
                    [
                        getattr(cfg.target_position_range, k)
                        for k in ("x", "y", "z")
                    ],
                    "object target",
                )
                command.target_pos[:] = torch.tensor(current["target_pos"])
                names = (
                    cfg.entity_names
                    if hasattr(cfg, "entity_names")
                    else [cfg.entity_name]
                )
                assert 0 <= current["target_selection"] < len(names)
                if hasattr(command, "target_selection"):
                    command.target_selection[:] = current["target_selection"]
                for name in names:
                    position, rotation, linear, angular = pose(name)
                    within(
                        position - env.scene.env_origins,
                        [
                            getattr(cfg.object_pose_range, k)
                            for k in ("x", "y", "z")
                        ],
                        "object position",
                    )
                    euler = torch.stack(euler_xyz_from_quat(rotation), dim=-1)
                    euler = (euler + np.pi) % (2 * np.pi) - np.pi
                    within(
                        euler,
                        [(0, 0), (0, 0), cfg.object_pose_range.yaw],
                        "object orientation",
                    )
                    np.testing.assert_allclose(linear, 0, atol=1e-6)
                    np.testing.assert_allclose(angular, 0, atol=1e-6)
                    env.scene[name].write_root_state_to_sim(
                        torch.cat(
                            (position, rotation, linear, angular), dim=-1
                        ),
                        env_ids=env_ids,
                    )

        command._resample_command = resample

    for index, name in enumerate(
        env.event_manager._mode_term_names.get("interval", [])
    ):
        term = env.event_manager._mode_term_cfgs["interval"][index]
        assert term.func.__name__ == "push_by_setting_velocity"

        def push(
            _env: Any,
            env_ids: Any,
            *,
            velocity_range: Any,
            asset_cfg: Any = None,
            _index: int = index,
            _name: str = name,
        ) -> None:
            assert (
                states[step - 1]["event_timers"][_name] <= env.step_dt + 1e-5
            ), "premature push"
            entity = "robot" if asset_cfg is None else asset_cfg.name
            velocity = torch.tensor(
                states[step]["push_velocities"][_name], dtype=torch.float32
            ).reshape(1, 6)
            within(
                velocity - env.scene[entity].data.root_link_vel_w,
                [
                    velocity_range.get(k, (0, 0))
                    for k in ("x", "y", "z", "roll", "pitch", "yaw")
                ],
                "push velocity",
            )
            env.scene[entity].write_root_link_velocity_to_sim(
                velocity, env_ids=env_ids
            )
            timer = states[step]["event_timers"][_name]
            within(
                timer,
                env.event_manager._mode_term_cfgs["interval"][
                    _index
                ].interval_range_s,
                "push duration",
            )
            env.event_manager._interval_term_time_left[_index][:] = timer

        term.func = push

    def prepare(index: int) -> None:
        nonlocal step
        import warp as wp

        step = index
        # The previous independent successor has already been recorded. Keep
        # command, contact and controller history; copy only integrator state.
        for name in ("qpos", "qvel"):
            target = getattr(env.sim.wp_data, name)
            value = integrator[name][step - 1].reshape(target.shape)
            wp.copy(target, wp.array(value, dtype=target.dtype, device="cpu"))
        env.sim.forward()
        target = env.sim.wp_data.qacc_warmstart
        value = integrator["warmstart"][step - 1].reshape(target.shape)
        wp.copy(target, wp.array(value, dtype=target.dtype, device="cpu"))
        env.sim.sense()

    def check() -> None:
        if command is not None:
            state = states[step]["command"]
            assert command.command_counter.item() == state["counter"]
            np.testing.assert_allclose(
                command.time_left.item(), state["time_left"], atol=1e-5
            )
            if hasattr(command, "motion"):
                assert command.time_steps.item() == state["time_steps"]
                np.testing.assert_allclose(
                    command.bin_failed_count.numpy(),
                    state["bin_failed_count"],
                    rtol=1e-5,
                    atol=1e-6,
                )
        for index, name in enumerate(
            env.event_manager._mode_term_names.get("interval", [])
        ):
            np.testing.assert_allclose(
                env.event_manager._interval_term_time_left[index].item(),
                states[step]["event_timers"][name],
                atol=1e-5,
            )

    return prepare, check


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
        # Compare the independently computed sensor values. Native observation
        # noise is checked separately against the pinned noise contract.
        for group in cfg.observations.values():
            group.enable_corruption = False
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
            prepare, check_samples = native_sample_checks(env, source)
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
                    # EnvPool's elapsed_step info uses int32; MJLab stores its
                    # internal episode counter as a Torch int64 tensor.
                    elapsed_step=env.episode_length_buf.numpy().astype(
                        np.int32
                    ),
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
                    frames.append(frame.copy())
                    frame_steps.append(step)

            record(0, 0, False, False)
            for step, action in enumerate(source["actions"], 1):
                prepare(step)
                obs, reward, terminated, truncated, _ = env.step(
                    torch.from_numpy(action[None])
                )
                check_samples()
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
                observation_dims=np.asarray(
                    json.dumps({
                        group: list(
                            zip(
                                env.observation_manager._group_obs_term_names[
                                    group
                                ],
                                [int(np.prod(shape)) for shape in shapes],
                                strict=True,
                            )
                        )
                        for group, shapes in env.observation_manager._group_obs_term_dim.items()
                    })
                ),
            )
            np.savez(args.output, allow_pickle=False, **result)
        finally:
            if renderer is not None:
                renderer.close()
            env.close()


if __name__ == "__main__":
    main()
