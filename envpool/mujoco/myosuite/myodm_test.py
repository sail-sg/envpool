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
"""Internal MyoSuite MyoDM TrackEnv native env tests."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Iterator

from envpool.python.glfw_context import preload_windows_gl_dlls

preload_windows_gl_dlls(strict=True)

import gymnasium
import mujoco
import numpy as np
from absl.testing import absltest

from envpool.mujoco.myosuite.metadata import MYOSUITE_DIRECT_ENTRIES
from envpool.mujoco.myosuite.native import (
    MyoDMTrackEnvSpec,
    MyoDMTrackGymnasiumEnvPool,
    MyoDMTrackPixelEnvSpec,
    MyoDMTrackPixelGymnasiumEnvPool,
)
from envpool.mujoco.myosuite.oracle_utils import (
    load_oracle_class,
    prepared_track_oracle_model_path,
)
from envpool.mujoco.myosuite.paths import myosuite_asset_root

_TRACK_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["class_name"] == "TrackEnv"
)
_TRACK_ALIGN_IDS = (
    "MyoHandAlarmclockFixed-v0",
    "MyoHandAirplaneFixed-v0",
    "MyoHandAirplaneRandom-v0",
    "MyoHandAirplaneFly-v0",
    "MyoHandPyramidmediumRandom-v0",
)
_TRACK_REPRESENTATIVE_IDS = (
    "MyoHandAirplaneFixed-v0",
    "MyoHandAirplaneRandom-v0",
    "MyoHandAirplaneFly-v0",
)
_ALIGNMENT_STEPS = 32
_TRACK_ALIGNMENT_OBS_TOLERANCE = {
    # The random airplane track reaches a mesh contact transition around step 8.
    # The XML, reset integration state, ctrl, act, and a pure Python mujoco
    # replay are bitwise with the official oracle; only EnvPool's Bazel-built
    # MuJoCo binary diverges from the pip MuJoCo binary in contact qacc by
    # ~2.4e-3, which accumulates to ~1.1e-5 in qvel. Keep this tolerance scoped
    # to that contact-heavy oracle check instead of widening all MyoDM tracks.
    "MyoHandAirplaneRandom-v0": 2e-5,
}


@dataclass(frozen=True)
class _TrackReferenceSample:
    """One official TrackEnv reference sample consumed during a rollout."""

    time: float
    robot: np.ndarray
    robot_vel: np.ndarray | None
    object: np.ndarray


def _entry(env_id: str) -> dict[str, Any]:
    return next(
        entry for entry in MYOSUITE_DIRECT_ENTRIES if entry["id"] == env_id
    )


def _make_env(config: tuple[Any, ...], pool_type: type, spec_type: type) -> Any:
    return pool_type(spec_type(config))


def _seeded_actions(
    shape: tuple[int, ...], steps: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        rng.uniform(-0.9, 0.9, size=shape).astype(np.float32)
        for _ in range(steps)
    ]


def _track_obs_tolerance(env_id: str) -> float:
    return _TRACK_ALIGNMENT_OBS_TOLERANCE.get(env_id, 1e-6)


def _assert_rollouts_match(
    case: absltest.TestCase,
    env0: Any,
    env1: Any,
    actions: list[np.ndarray],
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> None:
    obs0, info0 = env0.reset()
    obs1, info1 = env1.reset()
    np.testing.assert_allclose(obs0, obs1, atol=atol, rtol=rtol)
    case.assertEqual(
        info0["elapsed_step"].tolist(), info1["elapsed_step"].tolist()
    )
    for action in actions:
        obs0, reward0, terminated0, truncated0, info0 = env0.step(action)
        obs1, reward1, terminated1, truncated1, info1 = env1.step(action)
        np.testing.assert_allclose(obs0, obs1, atol=atol, rtol=rtol)
        np.testing.assert_allclose(reward0, reward1, atol=atol, rtol=rtol)
        np.testing.assert_array_equal(terminated0, terminated1)
        np.testing.assert_array_equal(truncated0, truncated1)
        case.assertEqual(
            info0["elapsed_step"].tolist(), info1["elapsed_step"].tolist()
        )
        if terminated0[0] or truncated0[0]:
            break


def _replace_all(text: str, old: str, new: str) -> str:
    return text.replace(old, new)


@contextmanager
def _edited_track_model(model_path: str, object_name: str) -> Iterator[Path]:
    asset_root = myosuite_asset_root()
    source_model = asset_root / model_path
    object_xml = source_model.read_text()
    tabletop_xml = (
        asset_root / "envs/myo/assets/hand/myohand_tabletop.xml"
    ).read_text()
    hand_assets_xml = (
        asset_root / "simhive/myo_sim/hand/assets/myohand_assets.xml"
    ).read_text()
    myo_sim_root = asset_root / "simhive/myo_sim"
    myo_sim_root_str = str(myo_sim_root)

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        hand_assets_tmp = tmp_dir / "myohand_assets.xml"
        tabletop_tmp = tmp_dir / "myohand_tabletop.xml"
        object_tmp = tmp_dir / "myohand_object.xml"

        hand_assets_xml = _replace_all(
            hand_assets_xml,
            'meshdir=".." texturedir=".."',
            f'meshdir="{myo_sim_root_str}" texturedir="{myo_sim_root_str}"',
        )
        hand_assets_tmp.write_text(hand_assets_xml)

        tabletop_xml = _replace_all(
            tabletop_xml,
            "../../../../simhive/myo_sim/hand/assets/myohand_assets.xml",
            str(hand_assets_tmp),
        )
        tabletop_xml = _replace_all(
            tabletop_xml,
            "../../../../simhive/furniture_sim/simpleTable/simpleTable_asset.xml",
            str(
                asset_root
                / "simhive/furniture_sim/simpleTable/simpleTable_asset.xml"
            ),
        )
        tabletop_xml = _replace_all(
            tabletop_xml,
            "../../../../simhive/myo_sim/hand/assets/myohand_body.xml",
            str(asset_root / "simhive/myo_sim/hand/assets/myohand_body.xml"),
        )
        tabletop_xml = _replace_all(
            tabletop_xml,
            "../../../../simhive/furniture_sim/simpleTable/simpleGraniteTable_body.xml",
            str(
                asset_root
                / "simhive/furniture_sim/simpleTable/simpleGraniteTable_body.xml"
            ),
        )
        tabletop_xml = _replace_all(
            tabletop_xml,
            'meshdir="../../../../simhive/myo_sim/" texturedir="../../../../simhive/myo_sim/"',
            f'meshdir="{myo_sim_root_str}" texturedir="{myo_sim_root_str}"',
        )
        tabletop_tmp.write_text(tabletop_xml)

        object_xml = _replace_all(object_xml, "OBJECT_NAME", object_name)
        object_xml = _replace_all(
            object_xml, "myohand_tabletop.xml", str(tabletop_tmp)
        )
        object_xml = _replace_all(
            object_xml,
            "../../../../simhive/object_sim/common.xml",
            str(asset_root / "simhive/object_sim/common.xml"),
        )
        object_xml = _replace_all(
            object_xml,
            f"../../../../simhive/object_sim/{object_name}/assets.xml",
            str(asset_root / f"simhive/object_sim/{object_name}/assets.xml"),
        )
        object_xml = _replace_all(
            object_xml,
            f"../../../../simhive/object_sim/{object_name}/body.xml",
            str(asset_root / f"simhive/object_sim/{object_name}/body.xml"),
        )
        object_tmp.write_text(object_xml)
        yield object_tmp


@cache
def _track_model(model_path: str, object_name: str) -> mujoco.MjModel:
    with _edited_track_model(model_path, object_name) as edited_model:
        return mujoco.MjModel.from_xml_path(str(edited_model))


def _dict_reference_init(
    reference: dict[str, Any], key: str, array_key: str
) -> np.ndarray:
    value = reference.get(key)
    if value is not None:
        return np.asarray(value, dtype=np.float64)
    base = np.asarray(reference[array_key], dtype=np.float64)
    if base.shape[0] == 2:
        return np.mean(base, axis=0)
    return base[0]


def _reference_config(reference: Any) -> dict[str, Any]:
    if isinstance(reference, str):
        reference_path = myosuite_asset_root() / reference
        with np.load(reference_path) as data:
            robot = np.asarray(data["robot"], dtype=np.float64)
            obj = np.asarray(data["object"], dtype=np.float64)
            robot_vel = (
                np.asarray(data["robot_vel"], dtype=np.float64)
                if "robot_vel" in data.files
                else None
            )
            robot_init = (
                np.asarray(data["robot_init"], dtype=np.float64)
                if "robot_init" in data.files
                else robot[0]
            )
            object_init = (
                np.asarray(data["object_init"], dtype=np.float64)
                if "object_init" in data.files
                else obj[0]
            )
        return {
            "reference_path": reference,
            "reference_time": [],
            "reference_robot": [],
            "reference_robot_vel": [],
            "reference_object": [],
            "reference_robot_init": robot_init.tolist(),
            "reference_object_init": object_init.tolist(),
            "robot_dim": int(robot.shape[1]),
            "object_dim": int(obj.shape[1]),
            "robot_horizon": int(robot.shape[0]),
            "object_horizon": int(obj.shape[0]),
            "reference_has_robot_vel": robot_vel is not None,
        }

    ref = dict(reference)
    time = np.asarray(ref["time"], dtype=np.float64)
    robot = np.asarray(ref["robot"], dtype=np.float64)
    obj = np.asarray(ref["object"], dtype=np.float64)
    robot_vel = (
        np.asarray(ref["robot_vel"], dtype=np.float64)
        if ref.get("robot_vel") is not None
        else None
    )
    robot_init = _dict_reference_init(ref, "robot_init", "robot")
    object_init = _dict_reference_init(ref, "object_init", "object")
    return {
        "reference_path": "",
        "reference_time": time.tolist(),
        "reference_robot": robot.reshape(-1).tolist(),
        "reference_robot_vel": []
        if robot_vel is None
        else robot_vel.reshape(-1).tolist(),
        "reference_object": obj.reshape(-1).tolist(),
        "reference_robot_init": robot_init.tolist(),
        "reference_object_init": object_init.tolist(),
        "robot_dim": int(robot.shape[1]),
        "object_dim": int(obj.shape[1]),
        "robot_horizon": int(robot.shape[0]),
        "object_horizon": int(obj.shape[0]),
        "reference_has_robot_vel": robot_vel is not None,
    }


def _track_config(
    env_id: str,
    *,
    overrides: dict[str, Any] | None = None,
    seed: int | None = None,
) -> tuple[tuple[Any, ...], type, type]:
    entry = _entry(env_id)
    kwargs = dict(entry["kwargs"])
    model = _track_model(kwargs["model_path"], kwargs["object_name"])
    reference = _reference_config(kwargs["reference"])
    obs_dim = (
        model.nq
        + model.nv
        + reference["robot_dim"]
        + (
            reference["robot_dim"]
            if reference["reference_has_robot_vel"]
            else 1
        )
        + 3
        + model.na
    )
    config = MyoDMTrackEnvSpec.gen_config(
        num_envs=1,
        batch_size=1,
        max_num_players=1,
        **({} if seed is None else {"seed": seed}),
        frame_skip=int(kwargs.get("frame_skip", 10)),
        model_path=str(kwargs["model_path"]),
        object_name=str(kwargs["object_name"]),
        reference_path=str(reference["reference_path"]),
        reference_time=list(reference["reference_time"]),
        reference_robot=list(reference["reference_robot"]),
        reference_robot_vel=list(reference["reference_robot_vel"]),
        reference_object=list(reference["reference_object"]),
        reference_robot_init=list(reference["reference_robot_init"]),
        reference_object_init=list(reference["reference_object_init"]),
        normalize_act=bool(kwargs.get("normalize_act", True)),
        obs_dim=obs_dim,
        qpos_dim=model.nq,
        qvel_dim=model.nv,
        act_dim=model.na,
        action_dim=model.nu,
        robot_dim=reference["robot_dim"],
        object_dim=reference["object_dim"],
        robot_horizon=reference["robot_horizon"],
        object_horizon=reference["object_horizon"],
        reference_has_robot_vel=bool(reference["reference_has_robot_vel"]),
        motion_start_time=float(kwargs.get("motion_start_time", 0.0)),
        motion_extrapolation=bool(kwargs.get("motion_extrapolation", True)),
        reward_pose_w=float(
            kwargs.get("weighted_reward_keys", {}).get("pose", 0.0)
        ),
        reward_object_w=float(
            kwargs.get("weighted_reward_keys", {}).get("object", 1.0)
        ),
        reward_bonus_w=float(
            kwargs.get("weighted_reward_keys", {}).get("bonus", 1.0)
        ),
        reward_penalty_w=float(
            kwargs.get("weighted_reward_keys", {}).get("penalty", -2.0)
        ),
        terminate_obj_fail=bool(
            kwargs.get(
                "Termimate_obj_fail", kwargs.get("terminate_obj_fail", True)
            )
        ),
        terminate_pose_fail=bool(
            kwargs.get(
                "Termimate_pose_fail", kwargs.get("terminate_pose_fail", False)
            )
        ),
    )
    if overrides:
        config = MyoDMTrackEnvSpec.gen_config(
            **dict(
                zip(MyoDMTrackEnvSpec._config_keys, config, strict=False),
                **overrides,
            )
        )
    return config, MyoDMTrackGymnasiumEnvPool, MyoDMTrackEnvSpec


def _track_pixel_config(env_id: str) -> tuple[tuple[Any, ...], type, type]:
    config, _, _ = _track_config(env_id)
    values = dict(zip(MyoDMTrackEnvSpec._config_keys, config, strict=False))
    pixel = MyoDMTrackPixelEnvSpec.gen_config(
        **values,
        render_width=64,
        render_height=64,
        render_camera_id=-1,
    )
    return pixel, MyoDMTrackPixelGymnasiumEnvPool, MyoDMTrackPixelEnvSpec


def _oracle_reference(reference: Any) -> Any:
    if isinstance(reference, str):
        return str(myosuite_asset_root() / reference)
    return {
        key: np.asarray(value, dtype=np.float64)
        for key, value in reference.items()
    }


@contextmanager
def _oracle_track_kwargs(env_id: str) -> Iterator[dict[str, Any]]:
    kwargs = dict(_entry(env_id)["kwargs"])
    with prepared_track_oracle_model_path() as model_path:
        yield {
            "object_name": kwargs["object_name"],
            "model_path": model_path,
            "reference": _oracle_reference(kwargs["reference"]),
        }


def _oracle_reset_sync(env: Any) -> tuple[np.ndarray, dict[str, Any]]:
    obs, _ = env.reset()
    unwrapped = env.unwrapped
    integration_state = unwrapped.sim.sim.get_state(
        int(mujoco.mjtState.mjSTATE_INTEGRATION)
    )
    return obs, {
        "test_reset_qpos": unwrapped.sim.data.qpos.copy().tolist(),
        "test_reset_qvel": unwrapped.sim.data.qvel.copy().tolist(),
        "test_reset_ctrl": unwrapped.sim.data.ctrl.copy().tolist(),
        "test_reset_act": (
            unwrapped.sim.data.act.copy().tolist()
            if unwrapped.sim.model.na > 0
            else []
        ),
        "test_reset_act_dot": (
            unwrapped.sim.data.act_dot.copy().tolist()
            if unwrapped.sim.model.na > 0
            else []
        ),
        "test_reset_qacc_warmstart": (
            unwrapped.sim.data.qacc_warmstart.copy().tolist()
        ),
        "test_reset_integration_state": integration_state.tolist(),
    }


@contextmanager
def _record_track_reference_samples(
    env: Any,
) -> Iterator[list[_TrackReferenceSample]]:
    unwrapped = env.unwrapped
    ref = getattr(unwrapped, "ref", None)
    if ref is None or not hasattr(ref, "get_reference"):
        yield []
        return
    original_get_reference = ref.get_reference
    samples: list[_TrackReferenceSample] = []

    def wrapped_get_reference(time: Any) -> Any:
        reference = original_get_reference(time)
        samples.append(
            _TrackReferenceSample(
                time=float(time),
                robot=np.asarray(reference.robot, dtype=np.float64).copy(),
                robot_vel=(
                    None
                    if reference.robot_vel is None
                    else np.asarray(
                        reference.robot_vel, dtype=np.float64
                    ).copy()
                ),
                object=np.asarray(reference.object, dtype=np.float64).copy(),
            )
        )
        return reference

    ref.get_reference = wrapped_get_reference
    try:
        yield samples
    finally:
        ref.get_reference = original_get_reference


def _track_reference_sync(
    samples: list[_TrackReferenceSample],
) -> dict[str, Any]:
    if not samples:
        return {}
    has_robot_vel = samples[0].robot_vel is not None
    return {
        "test_reference_time": [round(sample.time, 4) for sample in samples],
        "test_reference_robot": np.concatenate([
            sample.robot for sample in samples
        ]).tolist(),
        "test_reference_robot_vel": (
            []
            if not has_robot_vel
            else np.concatenate([
                sample.robot_vel for sample in samples
            ]).tolist()
        ),
        "test_reference_object": np.concatenate([
            sample.object for sample in samples
        ]).tolist(),
    }


class MyoDMTrackNativeTest(absltest.TestCase):
    """Verifies the native MyoDM TrackEnv surface."""

    def test_track_surface_count(self) -> None:
        """TrackEnv metadata should expose the full generated MyoDM surface."""
        self.assertLen(_TRACK_IDS, 190)

    def test_track_construction_covers_all_registered_ids(self) -> None:
        """Every TrackEnv id should construct and reset natively."""
        for env_id in _TRACK_IDS:
            with self.subTest(env_id=env_id):
                config, pool_type, spec_type = _track_config(env_id)
                env = _make_env(config, pool_type, spec_type)
                try:
                    obs, info = env.reset()
                    self.assertEqual(obs.shape[0], 1)
                    self.assertEqual(info["elapsed_step"].tolist(), [0])
                finally:
                    env.close()

    def test_track_native_determinism_representative_ids(self) -> None:
        """Fixed, random, and tracked references should be deterministic."""
        for env_id in _TRACK_REPRESENTATIVE_IDS:
            config, pool_type, spec_type = _track_config(env_id)
            env0 = _make_env(config, pool_type, spec_type)
            env1 = _make_env(config, pool_type, spec_type)
            model = _track_model(
                _entry(env_id)["kwargs"]["model_path"],
                _entry(env_id)["kwargs"]["object_name"],
            )
            actions = _seeded_actions((1, model.nu), steps=8, seed=173)
            with self.subTest(env_id=env_id):
                try:
                    _assert_rollouts_match(self, env0, env1, actions)
                finally:
                    env0.close()
                    env1.close()

    def test_track_alignment_representative_ids(self) -> None:
        """Fixed, random, and tracked references align with the official oracle."""
        cls = load_oracle_class("myosuite.envs.myo.myodm.myodm_v0", "TrackEnv")
        for env_id in _TRACK_ALIGN_IDS:
            with self.subTest(env_id=env_id):
                entry = _entry(env_id)
                with _oracle_track_kwargs(env_id) as oracle_kwargs:
                    oracle: Any | None = None
                    native: Any | None = None
                    try:
                        oracle = gymnasium.wrappers.TimeLimit(
                            cls(seed=123, **oracle_kwargs),
                            max_episode_steps=entry["max_episode_steps"],
                        )
                        obs0, sync = _oracle_reset_sync(oracle)
                        config, pool_type, spec_type = _track_config(
                            env_id, overrides=sync, seed=123
                        )
                        model = _track_model(
                            entry["kwargs"]["model_path"],
                            entry["kwargs"]["object_name"],
                        )
                        native = _make_env(config, pool_type, spec_type)
                        obs1, _ = native.reset()
                        np.testing.assert_allclose(
                            obs1, obs0[None, :], atol=1e-6, rtol=1e-6
                        )
                        actions = _seeded_actions(
                            (1, model.nu), steps=_ALIGNMENT_STEPS, seed=191
                        )
                        for step, action in enumerate(actions, start=1):
                            obs0, reward0, terminated0, truncated0, info0 = (
                                oracle.step(action[0])
                            )
                            obs1, reward1, terminated1, truncated1, info1 = (
                                native.step(action)
                            )
                            obs_tolerance = _track_obs_tolerance(env_id)
                            np.testing.assert_allclose(
                                obs1,
                                obs0[None, :],
                                atol=obs_tolerance,
                                rtol=obs_tolerance,
                                err_msg=f"{env_id} step {step} observation",
                            )
                            np.testing.assert_allclose(
                                reward1,
                                np.array([reward0]),
                                atol=1e-6,
                                rtol=1e-6,
                                err_msg=f"{env_id} step {step} reward",
                            )
                            np.testing.assert_array_equal(
                                terminated1, np.array([terminated0])
                            )
                            np.testing.assert_array_equal(
                                truncated1, np.array([truncated0])
                            )
                            if terminated0 or truncated0:
                                break
                    finally:
                        if native is not None:
                            native.close()
                        if oracle is not None:
                            oracle.close()

    def test_track_playback_reference_alignment(self) -> None:
        """Playback sync should consume repeated-time samples in oracle order."""
        env_id = "MyoHandPyramidmediumRandom-v0"
        entry = _entry(env_id)
        cls = load_oracle_class("myosuite.envs.myo.myodm.myodm_v0", "TrackEnv")
        with _oracle_track_kwargs(env_id) as oracle_kwargs:
            oracle: Any = None
            native: Any = None
            try:
                oracle = gymnasium.wrappers.TimeLimit(
                    cls(seed=123, **oracle_kwargs),
                    max_episode_steps=entry["max_episode_steps"],
                )
                with _record_track_reference_samples(oracle) as samples:
                    _, sync = _oracle_reset_sync(oracle)
                    _ = oracle.unwrapped.playback()
                    oracle_qpos = oracle.unwrapped.sim.data.qpos.copy()
                    oracle_qvel = oracle.unwrapped.sim.data.qvel.copy()
                self.assertEqual(
                    [round(sample.time, 4) for sample in samples], [0.0, 0.0]
                )
                sync["test_playback_reference"] = True
                sync.update(_track_reference_sync(samples))
                config, pool_type, spec_type = _track_config(
                    env_id, overrides=sync, seed=123
                )
                native = _make_env(config, pool_type, spec_type)
                _, _ = native.reset()
                model = _track_model(
                    entry["kwargs"]["model_path"],
                    entry["kwargs"]["object_name"],
                )
                zero = np.zeros((1, model.nu), dtype=np.float32)
                obs, _, _, _, _ = native.step(zero)
                qpos_dim = int(model.nq)
                qvel_dim = int(model.nv)
                native_qpos = np.asarray(obs)[0, :qpos_dim]
                native_qvel = np.asarray(obs)[0, qpos_dim : qpos_dim + qvel_dim]
                np.testing.assert_allclose(
                    native_qpos, oracle_qpos, atol=1e-9, rtol=1e-9
                )
                np.testing.assert_allclose(
                    native_qvel, oracle_qvel, atol=1e-9, rtol=1e-9
                )
            finally:
                if native is not None:
                    native.close()
                if oracle is not None:
                    oracle.close()

    def test_track_pixel_observation_smoke(self) -> None:
        """Track pixel wrappers should emit batched RGB observations."""
        config, pool_type, spec_type = _track_pixel_config(
            "MyoHandAirplaneFixed-v0"
        )
        env = _make_env(config, pool_type, spec_type)
        try:
            obs, _ = env.reset()
            self.assertEqual(obs.shape, (1, 3, 64, 64))
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
