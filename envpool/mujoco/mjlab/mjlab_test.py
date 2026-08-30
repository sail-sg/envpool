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
"""All-task native API, replay, render and default reset regression checks."""

import io
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from absl.testing import absltest, parameterized
from test_support import (
    actions,
    assert_observations,
    motion_file,
    public_components,
    task_options,
)

import envpool.mujoco.mjlab.registration  # noqa: F401
from envpool.mujoco.mjlab import TASKS
from envpool.registration import make_dm, make_gymnasium


class MjlabTest(parameterized.TestCase):
    """Verify the public API and native randomness without an oracle overwrite."""

    @parameterized.parameters(
        "stored", "deflated", "fortran", "big_endian", "float16", "float64"
    )
    def test_motion_encoding(self, encoding: str) -> None:
        """NumPy-compatible encodings must drive the same native trajectories."""
        folder = Path(self.create_tempdir().full_path)
        with np.load(motion_file(), allow_pickle=False) as source:
            arrays = {
                key: source[key][:17].copy()
                for key in source.files
                if source[key].ndim > 1
            }
        if encoding == "float16":
            arrays = {
                key: value.astype(np.float16).astype(np.float32)
                for key, value in arrays.items()
            }
        np.savez(folder / "reference.npz", **arrays)
        dtype = {"big_endian": ">f8", "float16": "f2", "float64": "f8"}.get(
            encoding, "f4"
        )
        encoded = {key: value.astype(dtype) for key, value in arrays.items()}
        if encoding == "fortran":
            encoded = {
                key: np.asfortranarray(value) for key, value in encoded.items()
            }
        writer = np.savez_compressed if encoding == "deflated" else np.savez
        writer(folder / "encoded.npz", **encoded)
        with ExitStack() as stack:
            pools = []
            for name in ("reference", "encoded"):
                pool = make_gymnasium(
                    "Mjlab-Tracking-Flat-Unitree-G1",
                    num_envs=1,
                    num_threads=1,
                    seed=17,
                    motion_file=str(folder / f"{name}.npz"),
                )
                stack.callback(pool.close)
                pools.append(pool)
            assert_observations(pools[0].reset()[0], pools[1].reset()[0])
            for control in actions(pools[0].action_space.shape[0], 128):
                left, right = [pool.step(control[None]) for pool in pools]
                assert_observations(left[0], right[0])
                for a, b in zip(left[1:4], right[1:4], strict=True):
                    np.testing.assert_array_equal(a, b)

    @parameterized.parameters(
        "missing",
        "wrong_shape",
        "nonfinite",
        "object",
        "checksum",
        "truncated",
        "oversized",
        "negative_shape",
        "nonexistent",
    )
    def test_invalid_motion(self, damage: str) -> None:
        """Reject malformed external motion data before starting a simulation."""
        folder = Path(self.create_tempdir().full_path)
        with np.load(motion_file(), allow_pickle=False) as source:
            arrays = {
                key: source[key][:17].copy()
                for key in source.files
                if source[key].ndim > 1
            }
        if damage == "missing":
            arrays.pop("body_ang_vel_w")
        elif damage == "wrong_shape":
            arrays["joint_pos"] = arrays["joint_pos"][:, :3]
        elif damage == "nonfinite":
            arrays["joint_pos"][0, 0] = np.nan
        elif damage == "object":
            arrays["joint_pos"] = arrays["joint_pos"].astype(object)
        path = folder / "motion.npz"
        np.savez(path, **arrays)
        if damage in ("oversized", "negative_shape"):
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(
                header,
                dict(
                    descr="<f4",
                    fortran_order=False,
                    shape=(2**62 if damage == "oversized" else -17, 29),
                ),
            )
            with ZipFile(path) as original:
                members = {
                    name: original.read(name) for name in original.namelist()
                }
            members["joint_pos.npy"] = header.getvalue()
            if damage == "negative_shape":
                members["joint_pos.npy"] += arrays["joint_pos"].tobytes()
            with ZipFile(path, "w") as archive:
                for name, content in members.items():
                    archive.writestr(name, content)
        elif damage == "checksum":
            with ZipFile(path) as archive:
                size = archive.infolist()[0].file_size
            data = bytearray(path.read_bytes())
            data[data.index(b"\x93NUMPY") + size - 1] ^= 1
            path.write_bytes(data)
        elif damage == "truncated":
            path.write_bytes(path.read_bytes()[:-23])
        elif damage == "nonexistent":
            path.unlink()
        with self.assertRaises(ValueError), ExitStack() as stack:
            pool = make_gymnasium(
                "Mjlab-Tracking-Flat-Unitree-G1",
                num_envs=1,
                num_threads=1,
                motion_file=str(path),
            )
            stack.callback(pool.close)
            pool.reset()

    @parameterized.named_parameters((task, task) for task in TASKS)
    def test_default_reset_randomization(self, task: str) -> None:
        """The public default API must not freeze goals or reset state (#432)."""
        sequences = []
        for seed in (11, 11, 43):
            with ExitStack() as stack:
                pool = make_gymnasium(
                    task,
                    num_envs=4,
                    num_threads=2,
                    seed=seed,
                    **task_options(task),
                )
                stack.callback(pool.close)
                resets = []
                for _ in range(8):
                    obs, info = pool.reset()
                    resets.append([
                        public_components(
                            task,
                            obs,
                            int(np.flatnonzero(info["env_id"] == slot)[0]),
                        )
                        for slot in range(4)
                    ])
                sequences.append(resets)
        for reset in range(8):
            for slot in range(4):
                assert_observations(
                    sequences[0][reset][slot],
                    sequences[1][reset][slot],
                    f"{task}, replay",
                )
        for component in sequences[0][0][0]:

            def differs(a: dict, b: dict, name: str = component) -> bool:
                return not np.array_equal(a[name], b[name])

            with self.subTest(component=component):
                self.assertTrue(
                    any(
                        differs(sequences[0][r][e], sequences[2][r][e])
                        for r in range(8)
                        for e in range(4)
                    ),
                    "different seeds",
                )
                self.assertTrue(
                    any(
                        differs(s[r][0], s[r][e])
                        for s in sequences
                        for r in range(8)
                        for e in range(1, 4)
                    ),
                    "parallel streams",
                )
                self.assertTrue(
                    any(
                        differs(s[0][e], s[r][e])
                        for s in sequences
                        for r in range(1, 8)
                        for e in range(4)
                    ),
                    "successive resets",
                )

    @parameterized.named_parameters((task, task) for task in TASKS)
    def test_rollout_and_render(self, task: str) -> None:
        """Replay across workers, observe autoresets, and render beyond reset."""
        kwargs = dict(
            num_envs=2,
            seed=[5, 9],
            max_episode_steps=96,
            render_mode="rgb_array",
            render_width=96,
            render_height=80,
            **task_options(task),
        )
        with ExitStack() as stack:
            left = make_gymnasium(task, num_threads=1, **kwargs)
            right = make_gymnasium(f"{task}-v0", num_threads=2, **kwargs)
            stack.callback(left.close)
            stack.callback(right.close)
            lo, li = left.reset()
            ro, ri = right.reset()
            controls = actions(left.action_space.shape[0], 195)
            endings = np.zeros(2, np.int32)
            for step, control in enumerate(controls):
                lp, rp = np.argsort(li["env_id"]), np.argsort(ri["env_id"])
                assert_observations(
                    {k: v[lp] for k, v in lo.items()},
                    {k: v[rp] for k, v in ro.items()},
                    f"{task}, step {step}",
                )
                for name in lo:
                    self.assertTrue(
                        left.observation_space[name].contains(lo[name][0]), name
                    )
                if step in (0, 32, 97, 194):
                    a, b = (
                        left.render(env_ids=[1, 0]),
                        right.render(env_ids=[1, 0]),
                    )
                    assert a is not None and b is not None
                    self.assertEqual(a.shape, (2, 80, 96, 3))
                    self.assertEqual(a.dtype, np.uint8)
                    np.testing.assert_array_equal(
                        a, b, err_msg=f"{task}, step {step}, worker replay"
                    )
                    np.testing.assert_array_equal(
                        a[:1], left.render(env_ids=[1])
                    )
                    self.assertGreater(int(a.max()) - int(a.min()), 20)
                control = np.repeat(control[None], 2, axis=0)
                lo, lr, lt, lx, li = left.step(control, env_id=li["env_id"])
                ro, rr, rt, rx, ri = right.step(control, env_id=ri["env_id"])
                lp, rp = np.argsort(li["env_id"]), np.argsort(ri["env_id"])
                for a, b in ((lr, rr), (lt, rt), (lx, rx)):
                    np.testing.assert_array_equal(a[lp], b[rp])
                endings += (lt | lx)[lp]
            self.assertTrue(np.all(endings >= 2))

    @parameterized.named_parameters((task, task) for task in TASKS)
    def test_dm_contract(self, task: str) -> None:
        """Validate dm_env observations and time-limit discount behavior."""
        with ExitStack() as stack:
            env = make_dm(
                task,
                num_envs=2,
                num_threads=2,
                max_episode_steps=3,
                **task_options(task),
            )
            stack.callback(env.close)
            timestep = env.reset()
            self.assertTrue(np.all(timestep.first()))
            for name, spec in env.observation_spec().obs.items():
                spec.validate(timestep.observation.obs[name][0])
            for _ in range(3):
                timestep = env.step(
                    np.zeros((2, *env.action_spec().shape), np.float32)
                )
            self.assertTrue(np.all(timestep.last()))
            np.testing.assert_array_equal(timestep.discount, np.ones(2))


if __name__ == "__main__":
    absltest.main()
