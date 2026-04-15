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

import importlib
import sys
import tempfile
import types
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any, Iterator

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
from envpool.mujoco.myosuite.paths import (
    myosuite_asset_root,
    resolve_workspace_path,
)
from envpool.python.glfw_context import preload_windows_gl_dlls

preload_windows_gl_dlls(strict=True)

_TRACK_IDS = tuple(
    entry["id"]
    for entry in MYOSUITE_DIRECT_ENTRIES
    if entry["class_name"] == "TrackEnv"
)
_TRACK_ALIGN_IDS = (
    "MyoHandAirplaneFixed-v0",
    "MyoHandAirplaneFly-v0",
)
_TRACK_REPRESENTATIVE_IDS = (
    "MyoHandAirplaneFixed-v0",
    "MyoHandAirplaneRandom-v0",
    "MyoHandAirplaneFly-v0",
)


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
            "reference_robot_init": [],
            "reference_object_init": [],
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
    env_id: str, *, overrides: dict[str, Any] | None = None
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


def _find_vendored_myosuite_root() -> Path:
    root = resolve_workspace_path(".")
    for candidate in (root, *root.parents):
        direct = candidate / "myosuite_src"
        if (direct / "myosuite/envs/myo/myodm/myodm_v0.py").exists() and (
            direct / "myosuite/simhive/object_sim"
        ).exists():
            return direct
        for source_path in candidate.rglob(
            "myosuite/envs/myo/myodm/myodm_v0.py"
        ):
            vendored_root = source_path.parents[4]
            if (vendored_root / "myosuite/simhive/object_sim").exists():
                return vendored_root
    raise FileNotFoundError("Unable to locate vendored myosuite source root")


@cache
def _prepare_oracle_myosuite_root() -> Path:
    source_root = _find_vendored_myosuite_root()
    external_root = source_root.parent
    temp_root = Path(tempfile.mkdtemp(prefix="envpool_myodm_oracle_"))
    package_root = temp_root / "myosuite"
    package_root.mkdir()

    for child in (source_root / "myosuite").iterdir():
        if child.name in {"simhive", "__pycache__"}:
            continue
        (package_root / child.name).symlink_to(child)

    simhive_root = package_root / "simhive"
    simhive_root.mkdir()
    simhive_sources = {
        "myo_sim": external_root / "myo_sim_src",
        "object_sim": external_root / "object_sim_src",
        "furniture_sim": external_root / "furniture_sim_src",
        "MPL_sim": external_root / "mpl_sim_src",
        "YCB_sim": external_root / "ycb_sim_src",
    }
    for name, fallback in simhive_sources.items():
        source = fallback
        if not source.exists():
            source = source_root / "myosuite" / "simhive" / name
        if source.exists():
            (simhive_root / name).symlink_to(source)
    return temp_root


def _install_flatten_dict_stub() -> None:
    if "flatten_dict" in sys.modules:
        return
    flatten_dict = types.ModuleType("flatten_dict")

    def flatten(
        mapping: dict[str, Any],
        reducer: str = "dot",
        keep_empty_types: tuple[type[Any], ...] = (),
    ) -> dict[tuple[str, ...], Any]:
        del reducer
        out: dict[tuple[str, ...], Any] = {}

        def rec(prefix: tuple[str, ...], value: Any) -> None:
            if isinstance(value, dict):
                if not value and dict in keep_empty_types:
                    out[prefix] = {}
                    return
                for key, child in value.items():
                    rec(prefix + (str(key),), child)
                return
            out[prefix] = value

        rec(tuple(), mapping)
        return out

    def unflatten(
        mapping: dict[tuple[str, ...], Any],
        splitter: str = "dot",
    ) -> dict[str, Any]:
        del splitter
        out: dict[str, Any] = {}
        for key, value in mapping.items():
            parts = (
                key if isinstance(key, tuple) else tuple(str(key).split("."))
            )
            cursor = out
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = value
        return out

    flatten_dict.__dict__["flatten"] = flatten
    flatten_dict.__dict__["unflatten"] = unflatten
    sys.modules["flatten_dict"] = flatten_dict


def _install_skvideo_stub() -> None:
    if "skvideo.io" in sys.modules:
        return
    skvideo = types.ModuleType("skvideo")
    skvideo_io = types.ModuleType("skvideo.io")
    skvideo.__dict__["io"] = skvideo_io
    sys.modules["skvideo"] = skvideo
    sys.modules["skvideo.io"] = skvideo_io


def _install_termcolor_stub() -> None:
    if "termcolor" in sys.modules:
        return
    termcolor = types.ModuleType("termcolor")

    def colored(text: Any, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        return str(text)

    def cprint(text: Any = "", *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        print(text)

    termcolor.__dict__["colored"] = colored
    termcolor.__dict__["cprint"] = cprint
    sys.modules["termcolor"] = termcolor


def _install_git_stub() -> None:
    if "git" in sys.modules:
        return
    git = types.ModuleType("git")

    class GitCommandError(RuntimeError):
        pass

    class _HeadCommit:
        hexsha = "stub"

    class _Head:
        commit = _HeadCommit()

    class _Git:
        def checkout(self, commit_hash: str) -> None:
            del commit_hash

    class Repo:
        def __init__(self, path: str) -> None:
            del path
            self.head = _Head()
            self.git = _Git()

        @classmethod
        def clone_from(cls, repo_url: str, clone_directory: str) -> "Repo":
            del repo_url
            return cls(clone_directory)

        def remote(self, name: str) -> Any:
            del name
            return types.SimpleNamespace(fetch=lambda: None)

    git.__dict__["GitCommandError"] = GitCommandError
    git.__dict__["Repo"] = Repo
    sys.modules["git"] = git


@cache
def _load_oracle_track_cls() -> Any:
    _install_flatten_dict_stub()
    _install_skvideo_stub()
    _install_termcolor_stub()
    _install_git_stub()
    sys.path.insert(0, str(_prepare_oracle_myosuite_root()))
    module = importlib.import_module("myosuite.envs.myo.myodm.myodm_v0")
    return module.TrackEnv


def _oracle_reference(reference: Any) -> Any:
    if isinstance(reference, str):
        return str(myosuite_asset_root() / reference)
    return {
        key: np.asarray(value, dtype=np.float64)
        for key, value in reference.items()
    }


def _oracle_track_kwargs(env_id: str) -> dict[str, Any]:
    kwargs = dict(_entry(env_id)["kwargs"])
    return {
        "object_name": kwargs["object_name"],
        "model_path": "/../assets/hand/myohand_object.xml",
        "reference": _oracle_reference(kwargs["reference"]),
    }


def _oracle_reset_sync(env: Any) -> tuple[np.ndarray, dict[str, Any]]:
    obs, _ = env.reset()
    unwrapped = env.unwrapped
    return obs, {
        "test_reset_qpos": unwrapped.sim.data.qpos.copy().tolist(),
        "test_reset_qvel": unwrapped.sim.data.qvel.copy().tolist(),
        "test_reset_act": (
            unwrapped.sim.data.act.copy().tolist()
            if unwrapped.sim.model.na > 0
            else []
        ),
        "test_reset_qacc_warmstart": (
            unwrapped.sim.data.qacc_warmstart.copy().tolist()
        ),
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
        """Fixed and tracked references should align with the official oracle."""
        cls = _load_oracle_track_cls()
        for env_id in _TRACK_ALIGN_IDS:
            with self.subTest(env_id=env_id):
                entry = _entry(env_id)
                oracle: Any = gymnasium.wrappers.TimeLimit(
                    cls(seed=123, **_oracle_track_kwargs(env_id)),
                    max_episode_steps=entry["max_episode_steps"],
                )
                obs0, sync = _oracle_reset_sync(oracle)
                config, pool_type, spec_type = _track_config(
                    env_id, overrides=sync
                )
                native = _make_env(config, pool_type, spec_type)
                try:
                    obs1, _ = native.reset()
                    np.testing.assert_allclose(
                        obs1, obs0[None, :], atol=1e-6, rtol=1e-6
                    )
                    model = _track_model(
                        entry["kwargs"]["model_path"],
                        entry["kwargs"]["object_name"],
                    )
                    actions = _seeded_actions((1, model.nu), steps=12, seed=191)
                    for action in actions:
                        obs0, reward0, terminated0, truncated0, _ = oracle.step(
                            action[0]
                        )
                        obs1, reward1, terminated1, truncated1, _ = native.step(
                            action
                        )
                        np.testing.assert_allclose(
                            obs1, obs0[None, :], atol=1e-6, rtol=1e-6
                        )
                        np.testing.assert_allclose(
                            reward1,
                            np.array([reward0]),
                            atol=1e-6,
                            rtol=1e-6,
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
                    native.close()
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
