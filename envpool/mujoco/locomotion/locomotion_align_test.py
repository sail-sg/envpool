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
from typing import Any

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
    check_reset_randomization,
)
from envpool.mujoco.locomotion.oracle import (
    EXAMPLE_MODULES,
    activate_oracle_context,
    make_oracle,
    oracle_observations,
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

    def assert_rewards(
        self, native: np.ndarray, oracle: Any, task: str, context: str
    ) -> None:
        """Compare rewards with only the measured scalar-math residuals."""
        expected = np.broadcast_to(oracle, native.shape)
        if (
            task == "rodent_two_touch"
            and platform.system() == "Linux"
            and platform.machine().lower() in {"x86_64", "amd64"}
        ):
            # NumPy's x86 SIMD exp and scalar glibc exp differ by one ULP.
            # The .01 * exp(-3 * distance) * 25 shaping term can amplify that
            # to two ULPs. Recomputing the oracle with math.exp reproduces the
            # native result exactly, including all three contact transitions.
            with self.subTest(context=context):
                np.testing.assert_array_max_ulp(native, expected, maxulp=2)
            return
        # Bowl's 3-vector norm differs by one ULP between BLAS and MuJoCo,
        # amplified by cancellation in 1 - (6 - distance) / 6. Tracking also
        # combines scalar exp and quaternion reductions (see tracking.cc).
        np.testing.assert_allclose(
            native,
            expected,
            rtol=0,
            atol=3e-15
            if task == "cmu_humanoid_tracking"
            else 1e-16
            if task == "rodent_escape_bowl"
            else 0,
            err_msg=context,
        )

    def assert_observations(
        self, native: dict, oracle: dict, task: str, context: str
    ) -> None:
        """Require exact state and pixels, with narrow derived-math residuals."""
        self.assertEqual(native.keys(), oracle.keys())
        for key in native:
            a, b = native[key], oracle[key]
            self.assertEqual(a.shape, b.shape, key)
            self.assertEqual(a.dtype, b.dtype, key)
            # Only derived Python/NumPy math differs: BLAS reduction order and
            # libm tanh/quaternion operations. Raw MuJoCo state/sensors, all
            # discrete values remain bitwise; camera residuals are separate.
            suffix = key.removeprefix("walker/")
            tolerance = 0.0
            if suffix in {"appendages_pos", "target", "origin"}:
                tolerance = 2e-15
            elif suffix == "sensors_torque":
                # NumPy's vector tanh and scalar libm differ by up to two
                # ULPs around 0.5 on Linux arm64 (one machine epsilon).
                tolerance = float(np.finfo(np.float64).eps)
            elif suffix.startswith("stats_") or suffix in {
                "velocimeter_control",
                "gyro_control",
            }:
                tolerance = 2e-14
            elif suffix.startswith("reference_") or suffix == "time_in_clip":
                tolerance = 1e-15
            elif suffix in {
                "team_goal_back_right",
                "team_goal_mid",
                "team_goal_front_left",
                "field_front_left",
                "opponent_goal_back_left",
                "opponent_goal_mid",
                "opponent_goal_front_right",
                "field_back_right",
            }:
                tolerance = 2e-14
            if key == "walker/egocentric_camera":
                assert_rgb_images(a, b, context)
            elif tolerance:
                np.testing.assert_allclose(
                    a, b, rtol=0, atol=tolerance, err_msg=f"{context}, {key}"
                )
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
    ) -> bool:
        official = make_oracle(task, seed)
        oracle_ts = official.reset()
        camera = (
            -1
            if camera_name is None
            else official.physics.model.name2id(camera_name, "camera")
        )
        # DMC requires its framebuffer dimensions to be set before rendering;
        # EnvPool grows its persistent framebuffer for larger public viewports.
        visual = official.physics.model.vis.global_
        visual.offwidth = max(visual.offwidth, width)
        visual.offheight = max(visual.offheight, height)
        env: Any = make_dm(
            f"dm_control/locomotion/{task}",
            seed=seed,
            num_envs=1,
            num_threads=1,
            render_mode="rgb_array",
            render_width=width,
            render_height=height,
            render_camera_id=camera,
        )
        timestep = env.reset()
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

        state = env._snapshot(include_model=True)
        native_model = mujoco.MjModel.from_binary_path(
            "native.mjb", {"native.mjb": state["model"]}
        )
        # Check the native geometry, inertias, actuators and heightfield before
        # allowing a reset-time state sync to eliminate bootstrap RNG details.
        for key in (
            "body_mass",
            "body_inertia",
            "body_pos",
            "geom_size",
            "geom_pos",
            "jnt_range",
            "actuator_gainprm",
            "actuator_biasprm",
            "hfield_data",
            "mat_rgba",
        ):
            np.testing.assert_array_equal(
                getattr(native_model, key),
                getattr(official.physics.model.ptr, key),
                err_msg=f"{task}, model {key}",
            )
        # Exactly one synchronization, immediately after reset. Neither engine
        # nor any task state is modified again during the action-only rollout.
        with official.physics.reset_context():
            official.physics.data.qpos[:] = state["qpos"]
            official.physics.data.qvel[:] = state["qvel"]
            official.physics.data.act[:] = state["act"]
        official.physics.data.qacc_warmstart[:] = state["warmstart"]
        frame = env.render()[0]
        activate_oracle_context(official)
        # Reset render also uses the synchronized state; no post-reset sync is
        # necessary for any later observation or frame.
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
            activate_oracle_context(official)
            oracle_ts = official.step(
                action if task.startswith("soccer_") else action[0]
            )
            timestep = env.step(action)
            self.assertEqual(timestep.reward.dtype, env.reward_spec().dtype)
            self.assertEqual(timestep.discount.dtype, env.discount_spec().dtype)
            self.assert_observations(
                timestep.observation.obs,
                oracle_observations(oracle_ts.observation),
                task,
                f"{task}, {pattern}, step {step}",
            )
            self.assert_rewards(
                timestep.reward,
                oracle_ts.reward,
                task,
                f"{task}, reward step {step}",
            )
            np.testing.assert_array_equal(
                timestep.discount,
                np.broadcast_to(oracle_ts.discount, (player_count,)),
            )
            self.assertEqual(bool(timestep.last()[0]), oracle_ts.last())
            state = env._snapshot()
            for key in ("qpos", "qvel", "act"):
                np.testing.assert_array_equal(
                    state[key],
                    getattr(official.physics.data, key),
                    err_msg=f"{task}, physics {key}, step {step}",
                )
            if step in (0, 3, 31, 127) or oracle_ts.last():
                frame = env.render()[0]
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
        starts = task._possible_starts
        probabilities = task._start_probabilities
        seeds: dict[int, int] = {}
        near_end = []
        for seed in range(10000):
            index = np.random.RandomState(seed).choice(
                len(starts), p=probabilities
            )
            clip, step = starts[index]
            seeds.setdefault(clip, seed)
            if task._dataset.end_steps[clip] - step == 16:
                near_end.append(seed)
            if len(seeds) == len(task._dataset.ids) and len(near_end) >= 12:
                break
        self.assertLen(seeds, len(task._dataset.ids))
        oracle.close()
        for clip, seed in sorted(seeds.items()):
            with self.subTest(clip=clip, seed=seed):
                self._run_episode("cmu_humanoid_tracking", "random", seed)
        for seed in near_end:
            if self._run_episode("cmu_humanoid_tracking", "reference", seed):
                break
        else:
            self.fail(
                "Reference-following actions never reached a clip boundary"
            )

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
        official.reset()
        env.reset()
        state = env._snapshot()
        model = official.physics.model.ptr
        fixture = mujoco.MjData(model)
        fixture.qpos[:] = state["qpos"]
        mujoco.mj_forward(model, fixture)
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
            activate_oracle_context(official)
            oracle_ts = official.step(action[0])
            timestep = env.step(action)
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
                np.testing.assert_array_equal(
                    snapshot[key], getattr(official.physics.data, key)
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
        ("timeout", -1.0, 35, 0.0, 4),
        ("early", 0.5, 35, -0.03, 3),
        ("success", -0.5, 65, -0.03, 2),
    )
    def test_two_touch_transitions(
        self, amplitude: float, duration: int, offset: float, outcome: int
    ) -> None:
        """Touch, release, and retouch using only actions after reset."""
        task = "rodent_two_touch"
        official = make_oracle(task)
        env: Any = make_dm(f"dm_control/locomotion/{task}", seed=0)
        official.reset()
        env.reset()
        state = env._snapshot()
        # Put the target by the hand once, before any external actions. This
        # makes the debounce, timing window and retargeting observable without
        # teleporting either body or target during the rollout.
        model = official.physics.model.ptr
        fixture = mujoco.MjData(model)
        fixture.qpos[:] = state["qpos"]
        fixture.qvel[:] = state["qvel"]
        mujoco.mj_forward(model, fixture)
        position = fixture.xpos[model.body("walker/hand_L").id].copy()
        position[0] += offset
        # Start from the XML pose; the nearby target is then reached by the
        # first pulse rather than already touching a hand at reset.
        state["qpos"] = model.qpos0.copy()
        state["qvel"] = np.zeros(model.nv)
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
            if 1 <= step < 8:
                action[0, 8:] = 1
            if 24 <= step < duration:
                action[0, 8:] = amplitude
            activate_oracle_context(official)
            oracle_ts = official.step(action[0])
            timestep = env.step(action)
            self.assert_observations(
                timestep.observation.obs,
                oracle_observations(oracle_ts.observation),
                task,
                f"two_touch, outcome {outcome}, step {step}",
            )
            self.assert_rewards(
                timestep.reward,
                oracle_ts.reward,
                task,
                f"two_touch, outcome {outcome}, reward step {step}",
            )
            np.testing.assert_array_equal(
                timestep.discount, [oracle_ts.discount]
            )
            self.assertEqual(bool(timestep.last()[0]), oracle_ts.last())
            snapshot = env._snapshot()
            for key in ("qpos", "qvel", "act"):
                np.testing.assert_array_equal(
                    snapshot[key],
                    getattr(official.physics.data, key),
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
        official.reset()
        env.reset()
        state = env._snapshot()
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
            oracle_ts = official.step(action)
            timestep = env.step(action)
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
                np.testing.assert_array_equal(
                    snapshot[key],
                    getattr(official.physics.data, key),
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
        # Enforce the one-reset-sync rule in the test hook itself.
        with self.assertRaisesRegex(RuntimeError, "only change reset state"):
            env._set_reset_state(0, qpos, qvel)
        official.close()
        env.close()


if __name__ == "__main__":
    absltest.main()
