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
"""Full-episode checks against dm_control 1.0.44, MuJoCo 3.11.0."""

import hashlib
import inspect
import os
import platform
from pathlib import Path
from typing import Any, cast

from envpool.mujoco.oracle import configure_mujoco_package_shared_lib
from envpool.python.glfw_context import preload_windows_gl_dlls

configure_mujoco_package_shared_lib(include_linux=True)
preload_windows_gl_dlls(strict=True)
if platform.system() == "Linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import mujoco
import numpy as np
from absl.testing import absltest, parameterized
from envpool.mujoco.locomotion.locomotion_envpool import TASKS

import envpool.mujoco.locomotion.registration  # noqa: F401
from envpool.mujoco.dmc.render_oracle import configure_macos_dm_control_renderer
from envpool.mujoco.locomotion.locomotion_test import (
    assert_egocentric_images,
    check_reset_randomization,
)
from envpool.mujoco.locomotion.oracle import (
    EXAMPLE_MODULES,
    activate_oracle_context,
    make_oracle,
    oracle_observations,
    reset_from_native,
    step_from_native,
)
from envpool.mujoco.render_test_utils import assert_rgb_images
from envpool.registration import make_dm

configure_macos_dm_control_renderer()


def randomized_components(pool: Any, env_id: int, obs: dict) -> dict:
    """Inspect the independent random components of the pinned factories."""
    task = pool.config["task_name"]
    state = pool._snapshot(env_id, include_model=True)
    model = mujoco.MjModel.from_binary_path(
        "reset.mjb", {"reset.mjb": state["model"]}
    )
    fields = {"spawn": state["qpos"]}
    if task.endswith(("run_walls", "run_gaps")):
        # These factories keep the initial walker pose fixed. Check geometry,
        # so camera noise or a changing texture cannot hide a frozen corridor.
        fields = {"layout": np.c_[model.geom_pos, model.geom_size]}
    elif task.endswith("forage"):
        # Sorting within each reward type excludes mere target-ID shuffling.
        fields["targets"] = np.asarray(
            sorted(
                (int(name.split("_")[1]), *model.body_pos[i])
                for i in range(model.nbody)
                if (name := model.body(i).name).startswith("target_")
            )
        )
        if "heterogeneous" in task:
            # Walls and the walker pose are fixed; goal locations and the
            # association of red/green textures with reward types are random.
            fields.pop("spawn")
            fields["target_colors"] = model.tex_data
        else:
            walls = state["maze"].replace("P", " ").replace("G", " ")
            fields["layout"] = np.frombuffer(walls.encode(), np.uint8)
    elif task == "cmu_humanoid_go_to_target":
        # The public target vector is relative to the walker: random spawn
        # positions must not mask an absolute goal stuck at a fixed location.
        fields["goal"] = model.site("target").pos
    elif task == "rodent_two_touch":
        fields["goal"] = model.geom("target_0_0/geom").pos
    elif task == "rodent_escape_bowl":
        fields["terrain"] = model.hfield_data
    elif task == "cmu_humanoid_tracking":
        fields["velocity"] = state["qvel"]
        fields["reference_clip"] = obs["walker/clip_id"]
        fields["reference_time"] = obs["walker/time_in_clip"]
    elif task.startswith("soccer_"):
        offset = model.jnt_qposadr[model.joint("soccer_ball/").id]
        fields = {
            "ball": state["qpos"][offset : offset + 3],
            "players": np.r_[
                state["qpos"][:offset], state["qpos"][offset + 7 :]
            ],
            "pitch": np.r_[
                model.site("field/lower").pos,
                model.site("field/upper").pos,
            ],
        }
        if task == "soccer_humanoid":
            hinges = model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE
            fields["pose"] = state["qpos"][model.jnt_qposadr[hinges]]
    else:
        raise AssertionError(f"Missing randomization components for {task}")

    fingerprints = {}
    for name, value in fields.items():
        value = np.asarray(value)
        if np.issubdtype(value.dtype, np.floating):
            value = value.copy()
            value[value == 0] = 0  # Signed zero is not state variation.
        # Do not retain dozens of complete heightfields and texture images.
        digest = hashlib.sha256(str((value.dtype, value.shape)).encode())
        digest.update(value.tobytes())
        fingerprints[name] = np.frombuffer(digest.digest(), np.uint8)
    return fingerprints


class LocomotionAlignTest(parameterized.TestCase):
    """Compare native rollouts to independent Composer task state."""

    @parameterized.named_parameters((task, task) for task in TASKS)
    def test_randomized_components(self, task: str) -> None:
        """Check real goals, layouts and spawns without oracle state sync."""
        check_reset_randomization(self, task, randomized_components)

    def assert_numerical(
        self, native: np.ndarray, oracle: Any, context: str
    ) -> None:
        """Allow roundoff in derived float64 math, independently of backend."""
        # Independent MuJoCo models and ordinary arithmetic differ at roundoff
        # level; acceleration sensors amplify velocity error by the timestep.
        # Match the float64 transition budget instead of requiring a particular
        # BLAS reduction or fused arithmetic path. Discrete values stay exact.
        np.testing.assert_allclose(
            native, oracle, rtol=1e-10, atol=1e-10, err_msg=context
        )

    def assert_observations(
        self, native: dict, oracle: dict, task: str, context: str
    ) -> None:
        """Compare derived values numerically and discrete observations exactly."""
        self.assertEqual(native.keys(), oracle.keys())
        for key in native:
            a, b = native[key], oracle[key]
            self.assertEqual(a.shape, b.shape, key)
            self.assertEqual(a.dtype, b.dtype, key)
            if key == "walker/egocentric_camera":
                assert_egocentric_images(a, b, task, context)
            elif np.issubdtype(a.dtype, np.floating):
                self.assert_numerical(a, b, f"{context}, {key}")
            else:
                np.testing.assert_array_equal(a, b, err_msg=f"{context}, {key}")

    @parameterized.named_parameters(
        (f"{task}_{pattern}", task, pattern)
        for task in TASKS
        for pattern in ("random", "sine")
    )
    def test_episode(self, task: str, pattern: str) -> None:
        """Run both implementations to the official episode end."""
        self._run_episode(task, pattern, 0 if pattern == "random" else 17)

    def _run_episode(
        self,
        task: str,
        pattern: str,
        seed: int,
        *,
        width: int = 128,
        height: int = 96,
        camera_name: str | None = None,
        env: Any = None,
        timestep: Any = None,
    ) -> bool:
        official = make_oracle(task, seed)
        owns_env = env is None
        if owns_env:
            env = make_dm(
                f"dm_control/locomotion/{task}",
                seed=seed,
                num_envs=1,
                num_threads=1,
                render_mode="rgb_array",
                render_width=width,
                render_height=height,
            )
            timestep = env.reset()
        env = cast(Any, env)
        state = reset_from_native(official, env)
        camera = (
            -1
            if camera_name is None
            else official.physics.model.name2id(camera_name, "camera")
        )
        specs = official.observation_spec()
        if isinstance(specs, list):
            specs = specs[0]
        for key, spec in specs.items():
            ours = env.observation_spec().obs[key]
            self.assertEqual(ours.shape, spec.shape, key)
            self.assertEqual(ours.dtype, spec.dtype, key)
        actions = official.action_spec()
        player_count = len(actions) if isinstance(actions, list) else 1
        action_spec = actions[0] if isinstance(actions, list) else actions
        reward_spec = official.reward_spec()
        if isinstance(reward_spec, list):
            reward_spec = reward_spec[0]
        self.assertEqual(env.reward_spec().dtype, reward_spec.dtype)
        self.assertEqual(
            env.discount_spec().dtype, official.discount_spec().dtype
        )
        self.assertEqual(env.action_spec().shape, action_spec.shape)
        self.assertEqual(env.action_spec().dtype, action_spec.dtype)
        np.testing.assert_array_equal(
            env.action_spec().minimum, action_spec.minimum
        )
        np.testing.assert_array_equal(
            env.action_spec().maximum, action_spec.maximum
        )

        visual = official.physics.model.vis.global_
        visual.offwidth = max(visual.offwidth, width)
        visual.offheight = max(visual.offheight, height)
        frame = env.render(camera_id=camera)[0]
        activate_oracle_context(official)
        # Both renderers now display the same checked native scene.
        reference_frame = official.physics.render(height, width, camera)
        if not np.array_equal(frame, reference_frame):
            from PIL import Image

            output = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp"))
            output.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(output / f"{task}-{pattern}-native.png")
            Image.fromarray(reference_frame).save(
                output / f"{task}-{pattern}-oracle.png"
            )
            (output / f"{task}-{pattern}-native.mjb").write_bytes(
                state["model"]
            )
            (output / f"{task}-{pattern}-oracle.mjb").write_bytes(
                official.physics.model.to_bytes()
            )

        assert_rgb_images(frame, reference_frame, f"{task}, reset render")
        random = np.random.RandomState(123)
        for step in range(2001):
            if pattern == "reference":
                tracking = official.task
                pose = tracking._clip_reference_features["joints"][
                    tracking._time_step + 1
                ]
                action = tracking._walker.cmu_pose_to_actuation(pose)[None]
            elif pattern == "random":
                action = random.uniform(
                    -0.1, 0.1, (player_count, *action_spec.shape)
                )
            else:
                action = 0.15 * np.sin(
                    np.arange(player_count * action_spec.shape[0]).reshape(
                        player_count, -1
                    )
                    * 0.37
                    + step * 0.21
                )
            # EnvPool render() can temporarily use the calling thread's GL
            # context. Invalidate DMC's cached binding before its next render.
            timestep, oracle_ts, state = step_from_native(official, env, action)
            self.assertEqual(timestep.reward.dtype, env.reward_spec().dtype)
            self.assertEqual(timestep.discount.dtype, env.discount_spec().dtype)
            self.assert_observations(
                timestep.observation.obs,
                oracle_observations(oracle_ts.observation),
                task,
                f"{task}, {pattern}, step {step}",
            )
            self.assert_numerical(
                timestep.reward,
                oracle_ts.reward,
                f"{task}, reward step {step}",
            )
            np.testing.assert_array_equal(
                timestep.discount,
                np.broadcast_to(oracle_ts.discount, (player_count,)),
            )
            self.assertEqual(bool(timestep.last()[0]), oracle_ts.last())
            state = env._snapshot()
            for key in ("qpos", "qvel", "act"):
                np.testing.assert_allclose(
                    state[key],
                    getattr(official.physics.data, key),
                    rtol=1e-10,
                    atol=1e-10,
                    err_msg=f"{task}, physics {key}, step {step}",
                )
            if step in (0, 3, 31, 127) or oracle_ts.last():
                frame = env.render(camera_id=camera)[0]
                activate_oracle_context(official)
                assert_rgb_images(
                    frame,
                    official.physics.render(height, width, camera),
                    f"{task}, render step {step}",
                )
            if oracle_ts.last():
                break
        else:
            self.fail(f"{task} did not reach the official episode end")
        # Exercise EnvPool's next-step autoreset without stepping the oracle a
        # second episode with an unsynchronized RNG/bootstrap state.
        self.assertTrue(env.step(np.zeros_like(action)).first()[0])
        clip_finished = bool(getattr(official.task, "_end_mocap", False))
        official.close()
        if owns_env:
            env.close()
        return clip_finished

    def test_tracking_side_camera(self) -> None:
        """Protect CGL MSAA readback ordering with the sensitive CMU camera."""
        self._run_episode(
            "cmu_humanoid_tracking",
            "sine",
            0,
            width=320,
            height=240,
            camera_name="walker/side",
        )

    def test_large_viewport(self) -> None:
        """Grow beyond the model's framebuffer without clipping the readback."""
        self._run_episode(
            "cmu_humanoid_go_to_target", "random", 0, width=1281, height=1025
        )

    def test_tracking_clip_coverage(self) -> None:
        """Exercise every WALK_TINY clip and the end-of-reference transition."""
        oracle = make_oracle("cmu_humanoid_tracking")
        task = oracle.task
        ends = np.asarray(task._dataset.end_steps) - np.asarray(
            task._dataset.start_steps
        )
        expected = set(range(len(task._dataset.ids)))
        oracle.close()
        seen = set()
        finished = False
        env: Any = make_dm(
            "dm_control/locomotion/cmu_humanoid_tracking",
            seed=0,
            num_envs=1,
            num_threads=1,
            render_mode="rgb_array",
            render_width=128,
            render_height=96,
        )
        try:
            for reset in range(10000):
                timestep = env.reset()
                state = env._snapshot()
                clip, start = state["reference_clip"], state["reference_start"]
                if clip not in seen:
                    with self.subTest(clip=clip, reset=reset):
                        self._run_episode(
                            "cmu_humanoid_tracking",
                            "random",
                            0,
                            env=env,
                            timestep=timestep,
                        )
                    seen.add(clip)
                elif not finished and ends[clip] - start == 16:
                    finished = self._run_episode(
                        "cmu_humanoid_tracking",
                        "reference",
                        0,
                        env=env,
                        timestep=timestep,
                    )
                if seen == expected and finished:
                    break
            self.assertEqual(seen, expected)
            self.assertTrue(
                finished, "Reference actions never reached a clip boundary"
            )
        finally:
            env.close()

    def test_upstream_registry_coverage(self) -> None:
        """Discover upstream factories and actually reset every public ID."""
        from dm_control.locomotion import soccer

        expected = {
            name
            for module in EXAMPLE_MODULES
            for name, function in inspect.getmembers(module, inspect.isfunction)
            if list(inspect.signature(function).parameters) == ["random_state"]
        }
        expected.update(
            f"soccer_{walker.name.lower()}" for walker in soccer.WalkerType
        )
        self.assertEqual(set(TASKS), expected)
        # Resolve and run both public names, not merely a copied allowlist.
        for task in expected:
            name = "".join(word.capitalize() for word in task.split("_"))
            env: Any = make_dm(f"Dmc{name}-v1", seed=0, max_episode_steps=1)
            timestep = env.reset()
            self.assertTrue(timestep.first()[0])
            env.close()

    @parameterized.named_parameters(
        (f"{task}_{scenario}", task, scenario)
        for task in (
            "cmu_humanoid_maze_forage",
            "rodent_maze_forage",
            "cmu_humanoid_heterogeneous_forage",
        )
        for scenario in (
            ("positive", "negative", "all")
            if "heterogeneous" in task
            else ("one", "all")
        )
    )
    def test_maze_collection(self, task: str, scenario: str) -> None:
        """Collect positive/negative targets and finish an entire maze."""
        official = make_oracle(task, seed=17)
        env: Any = make_dm(
            f"dm_control/locomotion/{task}",
            seed=17,
            render_mode="rgb_array",
            render_width=96,
            render_height=80,
        )
        env.reset()
        state = reset_from_native(official, env)
        model = official.physics.model.ptr
        # Placement only needs body transforms, not another multi-GB arena.
        fixture = official.physics.data.ptr
        fixture.qpos[:] = state["qpos"]
        mujoco.mj_kinematics(model, fixture)
        root = "walker/torso" if task.startswith("rodent_") else "walker/root"
        position = fixture.xpos[model.body(root).id]
        targets = official.task._active_targets
        selected = (
            [target for group in targets for target in group]
            if scenario == "all"
            else [targets[1 if scenario == "negative" else 0][0]]
        )
        geoms = {
            target.geom.full_identifier: position
            - fixture.xpos[model.geom(target.geom.full_identifier).bodyid[0]]
            for target in selected
        }
        env._set_reset_state(0, state["qpos"], state["qvel"], geoms)
        with official.physics.reset_context():
            official.physics.data.qpos[:] = state["qpos"]
            official.physics.data.qvel[:] = state["qvel"]
            official.physics.data.act[:] = state["act"]
            for name, local_position in geoms.items():
                official.physics.named.model.geom_pos[name] = local_position
        official.physics.data.qacc_warmstart[:] = 0
        rewards = []
        for step in range(12):
            action = np.full((1, *env.action_spec().shape), 0.05 * np.sin(step))
            timestep, oracle_ts, snapshot = step_from_native(
                official, env, action
            )
            self.assert_observations(
                timestep.observation.obs,
                oracle_observations(oracle_ts.observation),
                task,
                f"{task}, {scenario}, step {step}",
            )
            np.testing.assert_array_equal(timestep.reward, [oracle_ts.reward])
            np.testing.assert_array_equal(
                timestep.discount, [oracle_ts.discount]
            )
            self.assertEqual(bool(timestep.last()[0]), oracle_ts.last())
            rewards.append(float(timestep.reward[0]))
            snapshot = env._snapshot()
            for key in ("qpos", "qvel", "act"):
                np.testing.assert_allclose(
                    snapshot[key],
                    getattr(official.physics.data, key),
                    rtol=1e-10,
                    atol=1e-10,
                )
            frame = env.render()[0]
            activate_oracle_context(official)
            np.testing.assert_array_equal(
                frame, official.physics.render(80, 96)
            )
            if oracle_ts.last():
                break
        self.assertTrue(all(target.activated for target in selected))
        self.assertEqual(oracle_ts.last(), scenario == "all")
        self.assertLess(
            rewards[0], 0
        ) if scenario == "negative" else self.assertGreater(rewards[0], 0)
        for reward in rewards[1:]:
            self.assertEqual(reward, official.task._aliveness_reward)
        official.close()
        env.close()

    @parameterized.named_parameters(
        ("timeout", None, 4),
        ("early", 30, 3),
        ("success", 40, 2),
    )
    def test_two_touch_transitions(
        self, retouch_step: int | None, outcome: int
    ) -> None:
        """Touch, release, and retouch using only actions after reset."""
        task = "rodent_two_touch"
        official = make_oracle(task)
        env: Any = make_dm(f"dm_control/locomotion/{task}", seed=0)
        env.reset()
        state = reset_from_native(official, env)
        # Put the target by the hand once, before any external actions. This
        # makes the debounce, timing window and retargeting observable without
        # teleporting either body or target during the rollout.
        model = official.physics.model.ptr
        # Reuse the oracle arena while constructing the reset-time fixture.
        fixture = official.physics.data.ptr
        # Construct both the hand and target from the same canonical pose;
        # the fixture must not depend on a particular random spawn rotation.
        state["qpos"] = model.qpos0.copy()
        state["qvel"] = np.zeros(model.nv)
        fixture.qpos[:] = state["qpos"]
        mujoco.mj_kinematics(model, fixture)
        position = fixture.xpos[model.body("walker/hand_L").id].copy()
        position[0] -= 0.03
        geoms = {"target_0_0/geom": position}
        env._set_reset_state(0, state["qpos"], state["qvel"], geoms)
        with official.physics.reset_context():
            official.physics.data.qpos[:] = state["qpos"]
            official.physics.data.qvel[:] = state["qvel"]
            official.physics.data.act[:] = state["act"]
            official.physics.named.model.geom_pos["target_0_0/geom"] = position
        official.physics.data.qacc_warmstart[:] = 0
        events: list[int] = []
        for step in range(160):
            action = np.zeros((1, *env.action_spec().shape))
            if 1 <= step < 4 or (
                retouch_step is not None
                and retouch_step <= step < retouch_step + 10
            ):
                action[0, 8:] = 1
            elif step >= 4 and (retouch_step is None or step < retouch_step):
                action[0, 8:] = -1
            timestep, oracle_ts, snapshot = step_from_native(
                official, env, action
            )
            self.assert_observations(
                timestep.observation.obs,
                oracle_observations(oracle_ts.observation),
                task,
                f"two_touch, outcome {outcome}, step {step}",
            )
            self.assert_numerical(
                timestep.reward,
                oracle_ts.reward,
                f"two_touch, outcome {outcome}, reward step {step}",
            )
            np.testing.assert_array_equal(
                timestep.discount, [oracle_ts.discount]
            )
            self.assertEqual(bool(timestep.last()[0]), oracle_ts.last())
            snapshot = env._snapshot()
            for key in ("qpos", "qvel", "act"):
                np.testing.assert_allclose(
                    snapshot[key],
                    getattr(official.physics.data, key),
                    rtol=1e-10,
                    atol=1e-10,
                    err_msg=f"two_touch, {key}, step {step}",
                )
            touch = int(timestep.observation.obs["task_logic"].item())
            if not events or events[-1] != touch:
                events.append(touch)
        self.assertEqual(events[:4], [0, 1, outcome, 0])
        official.close()
        env.close()

    @parameterized.named_parameters(
        (f"{walker}_{scenario}", walker, scenario)
        for walker in ("boxhead", "ant", "humanoid")
        for scenario in ("goal", "multiturn", "offcourt", "field_box")
    )
    def test_soccer_transitions(self, walker: str, scenario: str) -> None:
        """Exercise goals, respawns, and throw-ins after one reset fixture."""
        task = f"soccer_{walker}"
        kwargs = dict(
            team_size=1,
            terminate_on_goal=scenario != "multiturn",
            enable_field_box=scenario == "field_box",
            disable_walker_contacts=True,
            keep_aspect_ratio=True,
        )
        official = make_oracle(task, **kwargs)
        env: Any = make_dm(f"dm_control/locomotion/{task}", seed=0, **kwargs)
        env.reset()
        state = reset_from_native(official, env)
        model = official.physics.model.ptr
        joint = model.joint("soccer_ball/").id
        address = model.jnt_qposadr[joint]
        qpos = state["qpos"]
        qvel = state["qvel"]
        if scenario in {"goal", "multiturn"}:
            qpos[address : address + 3] = [
                official.task.root_entity.home_goal.mid[0],
                0,
                0,
            ]
        else:
            qpos[address : address + 3] = [
                0,
                official.task.root_entity.field.upper[1] + 0.2,
                0.5,
            ]
        qvel[model.jnt_dofadr[joint] : model.jnt_dofadr[joint] + 6] = 0
        env._set_reset_state(0, qpos, qvel)
        with official.physics.reset_context():
            official.physics.data.qpos[:] = qpos
            official.physics.data.qvel[:] = qvel
            official.physics.data.act[:] = state["act"]
        official.physics.data.qacc_warmstart[:] = 0
        action = np.zeros((2, *env.action_spec().shape))
        saw_goal = False
        for step in range(20):
            timestep, oracle_ts, snapshot = step_from_native(
                official, env, action
            )
            self.assert_observations(
                timestep.observation.obs,
                oracle_observations(oracle_ts.observation),
                task,
                f"{task}, {scenario}, step {step}",
            )
            np.testing.assert_array_equal(timestep.reward, oracle_ts.reward)
            np.testing.assert_array_equal(
                timestep.discount, np.full(2, oracle_ts.discount)
            )
            self.assertEqual(bool(timestep.last()[0]), oracle_ts.last())
            saw_goal |= bool(np.any(timestep.reward))
            snapshot = env._snapshot()
            for key in ("qpos", "qvel", "act"):
                np.testing.assert_allclose(
                    snapshot[key],
                    getattr(official.physics.data, key),
                    rtol=1e-10,
                    atol=1e-10,
                    err_msg=f"{scenario}, {key}, step {step}",
                )
            if oracle_ts.last():
                self.assertEqual(scenario, "goal")
                break
        self.assertEqual(saw_goal, scenario in {"goal", "multiturn"})
        if scenario == "offcourt":
            # The first step detects off-court; the next action performs a
            # throw-in and zeros ball velocity before physics resumes.
            self.assertLess(
                abs(snapshot["qpos"][address + 1]),
                official.task.root_entity.field.upper[1],
            )
        if scenario == "field_box":
            self.assertFalse(oracle_ts.last())
        # Native fixtures remain immutable after reset; only oracle inputs sync.
        with self.assertRaisesRegex(RuntimeError, "only change reset state"):
            env._set_reset_state(0, qpos, qvel)
        official.close()
        env.close()


if __name__ == "__main__":
    absltest.main()
