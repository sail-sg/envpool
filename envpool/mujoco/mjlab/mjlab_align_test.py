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
"""Compare complete native episodes against unmodified official CPU MJLab."""

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
from absl.testing import absltest, parameterized

import envpool.mujoco.mjlab.registration  # noqa: F401
from envpool.mujoco.mjlab import TASKS
from envpool.mujoco.mjlab.test_support import actions, motion_file, task_options
from envpool.registration import make_gymnasium, make_spec


def oracle_command() -> list[str]:
    """Find the isolated oracle launcher without importing its dependencies."""
    suffix = "envpool/mujoco/mjlab/oracle_probe"
    candidates = []
    if manifest := os.environ.get("RUNFILES_MANIFEST_FILE"):
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            logical, _, physical = line.partition(" ")
            if logical.replace("\\", "/").endswith((suffix, suffix + ".exe")):
                candidates.append(Path(physical or logical))
    root = Path(os.environ["TEST_SRCDIR"])
    for workspace in (
        os.environ.get("TEST_WORKSPACE", "_main"),
        "_main",
        "envpool",
    ):
        candidates.extend(
            root / workspace / (suffix + extension)
            for extension in ("", ".exe")
        )
    for candidate in candidates:
        if candidate.is_file():
            return (
                [sys.executable, str(candidate)]
                if sys.platform == "win32" and candidate.suffix != ".exe"
                else [str(candidate)]
            )
    raise FileNotFoundError(
        "the pinned MJLab oracle launcher is missing from runfiles"
    )


def native_components(
    snapshot: dict, metadata: dict, motion: dict
) -> tuple[dict, dict, dict]:
    """Separate independently randomized semantic fields for the #432 gate."""
    state = json.loads(snapshot["task"])
    buffers = snapshot["physics"]

    def array(name: str, dtype: Any = np.float32) -> np.ndarray:
        return np.frombuffer(buffers[name], dtype).copy()

    qpos, qvel = array("data.qpos"), array("data.qvel")
    reset: dict[str, np.ndarray] = {}
    startup: dict[str, np.ndarray] = {}
    fixed: dict[str, np.ndarray] = {}
    name = metadata["id"]
    robot_name = "cartpole" if "Cartpole" in name else "robot"
    robot = metadata["entities"][robot_name]["indexing"]
    joints = np.asarray(robot["joint_q_adr"], dtype=int).ravel()
    if "Cartpole" in name:
        # MJLab 1.6 intentionally starts Swingup's slider at exactly zero.
        (fixed if "Swingup" in name else reset)["cart"] = qpos[:1]
        reset.update(
            pole=qpos[1:2], cart_velocity=qvel[:1], pole_velocity=qvel[1:2]
        )
    elif "Velocity" in name:
        root = qpos[np.asarray(robot["free_joint_q_adr"], dtype=int).ravel()]
        reset.update(position=root[:3] - state["origin"], orientation=root[3:7])
        command = state["command"]
        for i, axis in enumerate(("forward", "lateral", "turn")):
            reset["goal:" + axis] = np.array([command["command"][i]])
        reset["heading_target"] = np.array([command["heading_target"]])
        fixed["joints"] = qpos[joints]
    elif "Tracking" in name:
        command = state["command"]
        reset["reference"] = np.array(command["command"])
        # reset() advances the reference once after writing the sampled pose.
        # Subtract that sampled reference so a changing motion cannot conceal
        # missing independent pose/joint randomization.
        frame = command["time_steps"] - 1
        bodies = metadata["entities"]["robot"]["body_names"]
        reference_root = bodies.index(
            metadata["command"]["motion"]["body_names"][0]
        )
        root = qpos[np.asarray(robot["free_joint_q_adr"], dtype=int).ravel()]
        reset["position_noise"] = (
            root[:3]
            - motion["body_pos_w"][frame, reference_root]
            - state["origin"]
        )
        a = root[3:7].astype(np.float64)
        b = motion["body_quat_w"][frame, reference_root].astype(np.float64) * [
            1,
            -1,
            -1,
            -1,
        ]
        reset["orientation_noise"] = np.r_[
            a[0] * b[0] - a[1:] @ b[1:],
            a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:]),
        ]
        reset["joint_noise"] = qpos[joints] - motion["joint_pos"][frame]
    else:
        for axis, value in zip(
            "xyz", state["command"]["target_pos"], strict=True
        ):
            reset["goal:" + axis] = np.array([value])
        fixed["robot_joints"] = qpos[joints]
        for entity, info in metadata["entities"].items():
            if entity == "robot":
                continue
            indices = np.asarray(
                info["indexing"]["free_joint_q_adr"], dtype=int
            ).ravel()
            if indices.size:
                reset[entity + ":position"] = qpos[indices[:3]]
                reset[entity + ":orientation"] = qpos[indices[3:7]]
        if "Multi" in name:
            cubes = metadata["command"]["lift_height"]["entity_names"]
            selected = cubes[state["command"]["target_selection"]]
            geom = metadata["entities"][selected]["indexing"]["geom_ids"][0]
            # The cubes have distinct fixed colors: inspect the selected
            # object's appearance, not its ID or a command counter.
            reset["target_object_color"] = array("model.geom_rgba").reshape(
                -1, 4
            )[geom]

    for term_name, term in metadata["event"].items():
        function = term["func"]["callable"].rsplit(".", 1)[-1]
        if function not in {
            "encoder_bias",
            "body_com_offset",
            "geom_friction",
            "geom_rgba",
        }:
            continue
        target = startup if term["mode"] == "startup" else reset
        params = term["params"]
        asset = params.get("asset_cfg", {"name": "robot"})
        entity = asset["name"]
        kind = (
            "joint"
            if function == "encoder_bias"
            else "body"
            if function == "body_com_offset"
            else "geom"
        )
        mapping = np.asarray(
            metadata["entities"][entity]["indexing"][kind + "_ids"], dtype=int
        ).ravel()
        indices = asset.get(kind + "_ids", {"slice": [None, None, None]})
        if isinstance(indices, dict):
            indices = list(range(len(mapping)))[slice(*indices["slice"])]
        if function == "encoder_bias":
            values = np.asarray(state["entities"][entity]["encoder_bias"])[
                indices
            ]
            for i, value in enumerate(values):
                target[f"{term_name}:{i}"] = np.array([value])
            continue
        field = "body_ipos" if kind == "body" else function
        values = array("model." + field).reshape(
            -1, 4 if function == "geom_rgba" else 3
        )[mapping[indices]]
        ranges = params["ranges"]
        axes = params.get("axes")
        if axes is None:
            axes = (
                list(map(int, ranges))
                if isinstance(ranges, dict)
                else [0]
                if function == "geom_friction"
                else range(values.shape[1])
            )
        for axis in axes:
            limits = ranges[str(axis)] if isinstance(ranges, dict) else ranges
            if limits[0] == limits[1]:
                continue
            for i, value in enumerate(values[:, axis]):
                target[f"{term_name}:{axis}:{i}"] = np.array([value])
    if "terrain_state" in metadata:
        heights = array("model.hfield_data")
        addresses = array("model.hfield_adr", np.int32)
        rows, columns = (
            array("model.hfield_nrow", np.int32),
            array("model.hfield_ncol", np.int32),
        )
        for entry in metadata["terrain_state"]["random_heightfields"]:
            index = entry["hfield"]
            offset, count = addresses[index], rows[index] * columns[index]
            startup[f"terrain:{index}"] = heights[offset : offset + count]
    return reset, startup, fixed


def _terrain_cases() -> Iterator[tuple[str, str, int, int]]:
    for task, info in TASKS.items():
        if "Rough" not in task:
            continue
        base = Path(make_spec(task).config.base_path)
        metadata = json.loads(
            (
                base / "mujoco/mjlab/assets" / info["asset"] / "task.json"
            ).read_text()
        )
        columns = len(metadata["terrain_state"]["origins"][0])
        for column in range(columns):
            yield f"{task}_column{column}", task, columns, column


def _curriculum_cases() -> Iterator[tuple[str, str, int, bool]]:
    for task, info in TASKS.items():
        if "Velocity" not in task:
            continue
        base = Path(make_spec(task).config.base_path)
        metadata = json.loads(
            (
                base / "mujoco/mjlab/assets" / info["asset"] / "task.json"
            ).read_text()
        )
        stages = metadata["curriculum"]["command_vel"]["params"][
            "velocity_stages"
        ]
        for index, stage in enumerate(stages[1:]):
            yield f"{task}_step{stage['step']}", task, stage["step"], index == 0


class MjlabAlignTest(parameterized.TestCase):
    """Exercise every preset against the pinned official CPU implementation."""

    @parameterized.named_parameters((task, task) for task in TASKS)
    def test_native_randomization_components(self, task: str) -> None:
        """Never synchronize an oracle into these native default resets."""
        motion = {}
        if "Tracking" in task:
            with np.load(motion_file(), allow_pickle=False) as data:
                motion = {key: data[key] for key in data.files}
        fields = [
            "data.qpos",
            "data.qvel",
            "model.geom_friction",
            "model.geom_rgba",
            "model.body_ipos",
            "model.hfield_data",
            "model.hfield_adr",
            "model.hfield_nrow",
            "model.hfield_ncol",
        ]
        sequences = []
        metadata = None
        for seed in (11, 11, 43):
            with ExitStack() as stack:
                pool: Any = make_gymnasium(
                    task,
                    num_envs=4,
                    num_threads=2,
                    seed=seed,
                    **task_options(task),
                )
                stack.callback(pool.close)
                resets = []
                for _ in range(8):
                    pool.reset()
                    if metadata is None:
                        metadata = json.loads(
                            pool._snapshot(include_model=True)["metadata"]
                        )
                    resets.append([
                        native_components(
                            pool._snapshot(slot, fields=fields),
                            metadata,
                            motion,
                        )
                        for slot in range(4)
                    ])
                sequences.append(resets)
        for group in range(3):
            for field in sequences[0][0][0][group]:
                with self.subTest(group=group, field=field):
                    for reset in range(8):
                        for slot in range(4):
                            np.testing.assert_array_equal(
                                sequences[0][reset][slot][group][field],
                                sequences[1][reset][slot][group][field],
                            )

                    def differs(
                        a: tuple,
                        b: tuple,
                        group: int = group,
                        field: str = field,
                    ) -> bool:
                        # Reject mere roundoff as proof of randomization.
                        return bool(
                            np.max(np.abs(a[group][field] - b[group][field]))
                            > 1e-4
                        )

                    if group != 2:
                        self.assertTrue(
                            any(
                                differs(sequences[0][r][s], sequences[2][r][s])
                                for r in range(8)
                                for s in range(4)
                            ),
                            "different seeds",
                        )
                        self.assertTrue(
                            any(
                                differs(sequence[r][0], sequence[r][s])
                                for sequence in sequences
                                for r in range(8)
                                for s in range(1, 4)
                            ),
                            "parallel streams",
                        )
                    if group == 0:
                        self.assertTrue(
                            any(
                                differs(sequence[0][s], sequence[r][s])
                                for sequence in sequences
                                for r in range(1, 8)
                                for s in range(4)
                            ),
                            "successive resets",
                        )
                    else:
                        for sequence in sequences:
                            for reset in range(1, 8):
                                for slot in range(4):
                                    np.testing.assert_array_equal(
                                        sequence[0][slot][group][field],
                                        sequence[reset][slot][group][field],
                                    )
                    if group == 2:
                        for sequence in sequences:
                            for slot in range(4):
                                np.testing.assert_array_equal(
                                    sequence[0][slot][group][field],
                                    sequences[0][0][0][group][field],
                                )

    def run_oracle(self, folder: Path, source: Path | None = None) -> Path:
        """Run the official registry query or an independent complete rollout."""
        output = folder / ("oracle.npz" if source else "registry.json")
        cache = (
            Path(
                os.environ.get("MJLAB_ORACLE_CACHE", os.environ["TEST_TMPDIR"])
            )
            / "warp-cache"
        )
        cmd = oracle_command() + [
            "--output",
            str(output),
            "--cache",
            str(cache),
        ]
        cmd += ["--input", str(source)] if source else ["--registry"]
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        env.pop("WANDB_API_KEY", None)
        env["MPLCONFIGDIR"] = str(cache.parent / "matplotlib")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            timeout=600,
        )
        (folder / "oracle.log").write_text(result.stdout)
        self.assertEqual(result.returncode, 0, result.stdout[-12000:])
        return output

    def test_registry(self) -> None:
        """Fail when a builtin upstream task has no native registration."""
        folder = Path(self.create_tempdir().full_path)
        self.assertEqual(
            sorted(TASKS), json.loads(self.run_oracle(folder).read_text())
        )

    @parameterized.named_parameters(
        (f"{task}_seed{seed}", task, seed, False)
        for task in TASKS
        for seed in (17, 43)
    )
    def test_complete_episode(
        self, task: str, seed: int, short_motion: bool
    ) -> None:
        """Compare through termination or the preset time limit."""
        self.compare_episode(task, seed, short_motion)

    @parameterized.named_parameters(
        (task, task) for task in TASKS if "Tracking" in task
    )
    def test_motion_wrap(self, task: str) -> None:
        """Keep both implementations aligned across reference resampling."""
        self.compare_episode(task, 17, True)

    @parameterized.named_parameters(_terrain_cases())
    def test_terrain_column(self, task: str, columns: int, column: int) -> None:
        """Exercise each generated terrain, including the nonflat height scans."""
        self.compare_episode(task, 17, False, columns, column)

    @parameterized.named_parameters(_curriculum_cases())
    def test_curriculum(self, task: str, step: int, promote: bool) -> None:
        """Compare real resets and rollouts at each command-stage boundary."""
        self.compare_episode(task, 17, False, curriculum=(step, promote))

    def compare_episode(
        self,
        task: str,
        seed: int,
        short_motion: bool,
        num_envs: int = 1,
        env_slot: int = 0,
        curriculum: tuple[int, bool] | None = None,
        render_size: tuple[int, int] = (96, 80),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synchronize once after reset, then compare only action-driven steps."""
        folder = Path(self.create_tempdir().full_path)
        options = task_options(task)
        if short_motion:
            with np.load(motion_file(), allow_pickle=False) as data:
                np.savez_compressed(
                    folder / "short.npz",
                    **{
                        key: value[:17] if value.ndim > 1 else value
                        for key in data.files
                        for value in (data[key],)
                    },
                )
            options["motion_file"] = str(folder / "short.npz")
        with ExitStack() as stack:
            native: Any = make_gymnasium(
                task,
                num_envs=num_envs,
                batch_size=1,
                num_threads=1,
                seed=seed,
                render_mode="rgb_array",
                render_width=render_size[0],
                render_height=render_size[1],
                **options,
            )
            stack.callback(native.close)
            env_ids = np.array([env_slot], dtype=np.int32)
            obs, info = native.reset(env_id=env_ids)
            if curriculum is not None:
                native._prepare_reset(*curriculum)
            snapshot = native._snapshot(env_id=env_slot, include_model=True)
            if curriculum is not None:
                before = json.loads(snapshot["task"])
                obs, info = native.reset(env_id=env_ids)
                if "Rough" in task:
                    after = json.loads(native._snapshot()["task"])
                    # A real change of spawn position must accompany the level;
                    # checking only curriculum counters would miss a frozen map.
                    self.assertNotEqual(before["origin"], after["origin"])
            if num_envs > 1:
                metadata = json.loads(snapshot["metadata"])
                state = json.loads(snapshot["task"])
                level = state["command"]["terrain"]["level"]
                origin = np.asarray(
                    metadata["terrain_state"]["origins"][level][env_slot]
                )
                # Check the actual robot spawn, not a reported terrain ID: a
                # frozen column must not make seven oracle runs duplicate one.
                qpos = np.frombuffer(
                    snapshot["physics"]["data.qpos"], np.float32
                )
                root = np.asarray(
                    metadata["entities"]["robot"]["indexing"][
                        "free_joint_q_adr"
                    ],
                    dtype=int,
                ).ravel()[0]
                self.assertLess(
                    float(np.max(np.abs(qpos[root : root + 2] - origin[:2]))),
                    1.0,
                )
            controls = actions(
                native.action_space.shape[0],
                TASKS[task]["max_episode_steps"],
                seed,
            )
            render_steps = {0, 1, 17, 31, 64, 127, 255, 511}
            rows: dict[str, list[Any]] = {}
            frames, frame_steps = [], []
            motion_frames = []

            def record(
                step: int, reward: float, terminated: bool, truncated: bool
            ) -> None:
                state = native._snapshot(env_id=env_slot)
                values = {
                    f"obs:{key}": value.copy() for key, value in obs.items()
                }
                values.update(
                    reward=np.array(reward, np.float32),
                    terminated=np.array(terminated),
                    truncated=np.array(truncated),
                    elapsed_step=info["elapsed_step"].copy(),
                )
                for key in ("qpos", "qvel"):
                    values[key] = np.frombuffer(
                        state["physics"]["data." + key], np.float32
                    )[None].copy()
                for key, value in values.items():
                    rows.setdefault(key, []).append(value)
                if short_motion:
                    motion_frames.append(
                        json.loads(state["task"])["command"]["time_steps"]
                    )
                if step in render_steps or terminated or truncated:
                    frames.append(native.render(env_ids=env_ids)[0].copy())
                    frame_steps.append(step)

            record(0, 0, False, False)
            for step, control in enumerate(controls, 1):
                obs, reward, terminated, truncated, info = native.step(
                    control[None], env_id=env_ids
                )
                record(step, reward[0], terminated[0], truncated[0])
                if terminated[0] or truncated[0]:
                    break
            self.assertTrue(
                terminated[0] or truncated[0],
                "compare through the actual episode boundary",
            )
            if short_motion:
                self.assertTrue(
                    np.any(np.diff(motion_frames) < 0),
                    "the rollout must cross a motion-file boundary",
                )
            source = folder / "input.npz"
            initial = {
                "physics:" + key: np.frombuffer(value, np.uint8)
                for key, value in snapshot["physics"].items()
            }
            np.savez_compressed(
                source,
                **initial,
                model=np.frombuffer(snapshot["model"], np.uint8),
                state=snapshot["task"],
                task=task,
                motion_file=options.get("motion_file", ""),
                actions=controls,
                render_steps=np.array(frame_steps),
                reset_after_sync=curriculum is not None,
                render_size=np.array(render_size),
            )
            output = self.run_oracle(folder, source)
            with np.load(output, allow_pickle=False) as oracle:
                np.testing.assert_array_equal(
                    native.action_space.shape, oracle["action_shape"]
                )
                np.testing.assert_array_equal(
                    native.action_space.low, oracle["action_low"]
                )
                np.testing.assert_array_equal(
                    native.action_space.high, oracle["action_high"]
                )
                all_values = {
                    key: np.stack(value) for key, value in rows.items()
                }
                all_values["frames"] = np.stack(frames)
                all_values["frame_steps"] = np.array(frame_steps)
                if any(
                    not np.array_equal(value, oracle[key])
                    for key, value in all_values.items()
                ):
                    artifact = Path(
                        os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", folder)
                    ) / (
                        f"{task}-seed{seed}-short{short_motion}-slot{env_slot}"
                        f"-curriculum{curriculum[0] if curriculum else 'none'}"
                    )
                    artifact.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(artifact / "native.npz", **all_values)
                    shutil.copy2(source, artifact / "input.npz")
                    shutil.copy2(output, artifact / "oracle.npz")
                    shutil.copy2(folder / "oracle.log", artifact / "oracle.log")
                for key, values in all_values.items():
                    with self.subTest(field=key):
                        self.assertTrue(np.isfinite(values).all(), key)
                        np.testing.assert_array_equal(
                            values,
                            oracle[key],
                            err_msg=f"{task}, seed {seed}, {key}",
                        )
                return all_values["frames"], oracle["frames"].copy()


if __name__ == "__main__":
    absltest.main()
