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
"""Gymnasium-Robotics adapter backend."""

from __future__ import annotations

import contextlib
import functools
import keyword
import os
import pprint
import re
import sys
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

if sys.platform.startswith("linux") and "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")

import dm_env
import gymnasium
import gymnasium_robotics  # noqa: F401
import numpy as np
from dm_env import TimeStep

from envpool.python.data import (
    dm_spec_transform,
    to_namedtuple,
    to_nested_dict,
)
from envpool.python.glfw_context import try_ensure_mujoco_glfw_context
from envpool.python.protocol import ArraySpec, ThreadPoolArg

_INT32 = np.dtype(np.int32).type
_BOOL = np.dtype(np.bool_).type
_INVALID_DM_FIELD_CHARS = re.compile(r"\W|^(?=\d)")
_DEFAULT_CONFIG = {
    "num_envs": 1,
    "batch_size": 0,
    "num_threads": 0,
    "max_num_players": 1,
    "thread_affinity_offset": -1,
    "base_path": "",
    "seed": 0,
    "env_seed": [],
    "gym_reset_return_info": True,
    "max_episode_steps": 0,
    "reward_threshold": None,
    "gymnasium_task_id": "",
    "gymnasium_robotics_kwargs": {},
}


def _as_int32_env_ids(env_id: Any, default_env_ids: np.ndarray) -> np.ndarray:
    if env_id is None:
        return default_env_ids.copy()
    if isinstance(env_id, (int, np.integer)):
        return np.asarray([env_id], dtype=np.int32)
    env_ids = np.asarray(env_id, dtype=np.int32)
    if env_ids.ndim == 0:
        return env_ids.reshape(1)
    return env_ids


def _seed_for_env(seed: int) -> int:
    return int(np.uint32(seed))


def _box_spec(space: gymnasium.spaces.Box) -> ArraySpec:
    dtype = np.dtype(space.dtype).type
    if space.shape:
        return ArraySpec(
            dtype=dtype,
            shape=list(space.shape),
            bounds=(0, 0),
            element_wise_bounds=(
                np.asarray(space.low).tolist(),
                np.asarray(space.high).tolist(),
            ),
        )
    return ArraySpec(
        dtype=dtype,
        shape=[],
        bounds=(
            np.asarray(space.low, dtype=space.dtype).item(),
            np.asarray(space.high, dtype=space.dtype).item(),
        ),
        element_wise_bounds=([], []),
    )


def _discrete_spec(space: gymnasium.spaces.Discrete) -> ArraySpec:
    start = int(space.start)
    return ArraySpec(
        dtype=np.dtype(space.dtype).type,
        shape=[],
        bounds=(start, start + int(space.n) - 1),
        element_wise_bounds=([], []),
        is_discrete=True,
    )


def _space_spec(space: gymnasium.Space[Any]) -> ArraySpec:
    if isinstance(space, gymnasium.spaces.Box):
        return _box_spec(space)
    if isinstance(space, gymnasium.spaces.Discrete):
        return _discrete_spec(space)
    if isinstance(space, gymnasium.spaces.MultiBinary):
        return ArraySpec(
            dtype=np.dtype(space.dtype).type,
            shape=[int(s) for s in np.atleast_1d(space.shape)],
            bounds=(0, 1),
            element_wise_bounds=([], []),
            is_discrete=True,
        )
    if isinstance(space, gymnasium.spaces.MultiDiscrete):
        return ArraySpec(
            dtype=np.dtype(space.dtype).type,
            shape=list(np.asarray(space.nvec).shape),
            bounds=(0, 0),
            element_wise_bounds=(
                np.zeros_like(space.nvec, dtype=np.int64).tolist(),
                (np.asarray(space.nvec, dtype=np.int64) - 1).tolist(),
            ),
            is_discrete=True,
        )
    raise TypeError(f"Unsupported Gymnasium-Robotics space: {space!r}")


def _flatten_space(
    space: gymnasium.Space[Any],
    prefix: str,
) -> dict[str, ArraySpec]:
    if isinstance(space, gymnasium.spaces.Dict):
        ret = {}
        for key, child in space.spaces.items():
            separator = "." if ":" in prefix else ":"
            ret.update(_flatten_space(child, f"{prefix}{separator}{key}"))
        return ret
    return {prefix: _space_spec(space)}


def _flatten_info_specs(info: Mapping[str, Any]) -> dict[str, ArraySpec]:
    ret: dict[str, ArraySpec] = {
        "info:env_id": ArraySpec(
            dtype=_INT32,
            shape=[],
            bounds=(0, 2**31 - 1),
            element_wise_bounds=([], []),
            is_discrete=True,
        ),
        "info:elapsed_step": ArraySpec(
            dtype=_INT32,
            shape=[],
            bounds=(0, 2**31 - 1),
            element_wise_bounds=([], []),
            is_discrete=True,
        ),
    }
    for key, value in info.items():
        arr = np.asarray(value)
        if arr.dtype == np.dtype("O"):
            continue
        if key in {"env_id", "elapsed_step"}:
            continue
        if arr.dtype == np.dtype(np.bool_):
            ret[f"info:{key}"] = ArraySpec(
                dtype=_BOOL,
                shape=list(arr.shape),
                bounds=(0, 1),
                element_wise_bounds=([], []),
                is_discrete=True,
            )
        elif np.issubdtype(arr.dtype, np.integer):
            if arr.shape:
                ret[f"info:{key}"] = ArraySpec(
                    dtype=np.dtype(arr.dtype).type,
                    shape=list(arr.shape),
                    bounds=(0, 0),
                    element_wise_bounds=(
                        arr.tolist(),
                        arr.tolist(),
                    ),
                )
            else:
                value_int = int(arr.item())
                ret[f"info:{key}"] = ArraySpec(
                    dtype=np.dtype(arr.dtype).type,
                    shape=[],
                    bounds=(value_int, value_int),
                    element_wise_bounds=([], []),
                )
        else:
            dtype = np.dtype(arr.dtype).type
            shape = list(arr.shape)
            if shape:
                ret[f"info:{key}"] = ArraySpec(
                    dtype=dtype,
                    shape=shape,
                    bounds=(0, 0),
                    element_wise_bounds=(
                        np.asarray(arr).tolist(),
                        np.asarray(arr).tolist(),
                    ),
                )
            else:
                value_float = float(arr.item())
                ret[f"info:{key}"] = ArraySpec(
                    dtype=dtype,
                    shape=[],
                    bounds=(value_float, value_float),
                    element_wise_bounds=([], []),
                )
    return ret


def _make_upstream_env(
    gymnasium_task_id: str,
    max_episode_steps: int,
    kwargs: Any,
) -> gymnasium.Env:
    if hasattr(kwargs, "_asdict"):
        make_kwargs = dict(kwargs._asdict())
    else:
        make_kwargs = dict(kwargs)
    make_kwargs["render_mode"] = "rgb_array"
    if max_episode_steps > 0:
        make_kwargs["max_episode_steps"] = max_episode_steps
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with np.errstate(all="ignore"):
            return gymnasium.make(gymnasium_task_id, **make_kwargs)


def _stack_space_values(
    values: Sequence[Any],
    space: gymnasium.Space[Any],
) -> Any:
    if isinstance(space, gymnasium.spaces.Dict):
        return {
            key: _stack_space_values(
                [value[key] for value in values],
                child,
            )
            for key, child in space.spaces.items()
        }
    return np.stack(
        [np.asarray(value, dtype=space.dtype) for value in values],
        axis=0,
    )


def _stack_info_values(values: Sequence[Any]) -> np.ndarray:
    arrays = [np.asarray(value) for value in values]
    if all(
        arr.dtype != np.dtype("O") and arr.shape == arrays[0].shape
        for arr in arrays
    ):
        return np.stack(arrays, axis=0)
    return np.asarray(values, dtype=object)


def _stack_infos(
    infos: Sequence[Mapping[str, Any]],
    env_ids: np.ndarray,
    elapsed_step: np.ndarray,
) -> dict[str, np.ndarray]:
    keys = sorted({key for info in infos for key in info})
    ret = {
        key: _stack_info_values([info.get(key) for info in infos])
        for key in keys
    }
    ret["env_id"] = np.asarray(env_ids, dtype=np.int32).copy()
    ret["elapsed_step"] = np.asarray(elapsed_step, dtype=np.int32).copy()
    return ret


def _action_at(
    action: Any,
    index: int,
    space: gymnasium.Space[Any],
) -> Any:
    if isinstance(space, gymnasium.spaces.Dict):
        if not isinstance(action, Mapping):
            raise TypeError("Dict action is required for Dict action_space")
        return {
            key: _action_at(action[key], index, child)
            for key, child in space.spaces.items()
        }
    return np.asarray(action, dtype=space.dtype)[index]


def _build_dm_observation(obs: Any, info: Mapping[str, Any]) -> tuple:
    if isinstance(obs, Mapping):
        payload = dict(obs)
    else:
        payload = {"obs": obs}
    payload.update(info)
    return to_namedtuple(
        "State",
        _sanitize_dm_namedtuple_tree(to_nested_dict(payload)),
    )


def _sanitize_dm_namedtuple_name(name: str) -> str:
    safe_name = _INVALID_DM_FIELD_CHARS.sub("_", name)
    if keyword.iskeyword(safe_name):
        safe_name = f"{safe_name}_"
    return safe_name or "_"


def _sanitize_dm_namedtuple_tree(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    ret: dict[str, Any] = {}
    for key, child in value.items():
        safe_key = _sanitize_dm_namedtuple_name(str(key))
        suffix = 1
        deduped_key = safe_key
        while deduped_key in ret:
            suffix += 1
            deduped_key = f"{safe_key}_{suffix}"
        ret[deduped_key] = _sanitize_dm_namedtuple_tree(child)
    return ret


def _dm_timestep(
    obs: Any,
    reward: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    info: Mapping[str, Any],
) -> TimeStep:
    done = np.logical_or(terminated, truncated)
    return TimeStep(
        step_type=np.where(
            done,
            dm_env.StepType.LAST,
            dm_env.StepType.MID,
        ).astype(np.int32),
        reward=np.asarray(reward, dtype=np.float64),
        discount=np.where(
            terminated,
            0.0,
            1.0,
        ).astype(np.float64),
        observation=_build_dm_observation(obs, info),
    )


class GymnasiumRoboticsEnvSpec:
    """EnvSpec wrapper around a Gymnasium-Robotics task."""

    _config_keys: ClassVar[list[str]] = list(_DEFAULT_CONFIG)
    _default_config_values: ClassVar[tuple[Any, ...]] = tuple(
        _DEFAULT_CONFIG.values()
    )

    @staticmethod
    def gen_config(**kwargs: Any) -> dict[str, Any]:
        """Merge common EnvPool config with upstream Gymnasium kwargs."""
        config = dict(_DEFAULT_CONFIG)
        robotics_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in config:
                config[key] = value
            else:
                robotics_kwargs[key] = value
        config["gymnasium_robotics_kwargs"] = dict(
            cast(dict[str, Any], config["gymnasium_robotics_kwargs"]),
            **robotics_kwargs,
        )
        return config

    def __init__(self, config: Mapping[str, Any]):
        """Build specs from the upstream environment spaces."""
        self._config_dict = dict(config)
        self._config_values = tuple(
            self._config_dict[key] for key in self._config_keys
        )

        env = _make_upstream_env(
            self.config.gymnasium_task_id,
            self.config.max_episode_steps,
            self.config.gymnasium_robotics_kwargs,
        )
        try:
            self._gymnasium_observation_space = env.observation_space
            self._gymnasium_action_space = env.action_space
            _, reset_info = env.reset(seed=_seed_for_env(self.config.seed))
            _, _, _, _, step_info = env.step(env.action_space.sample())
            self._state_specs = {
                **_flatten_space(
                    self._gymnasium_observation_space,
                    "obs",
                ),
                **_flatten_info_specs({**reset_info, **step_info}),
            }
            self._action_specs = _flatten_space(
                self._gymnasium_action_space,
                "action",
            )
        finally:
            env.close()

    @functools.cached_property
    def config(self) -> Any:
        """Return the concrete task config as an attribute tuple."""
        return to_namedtuple("Config", self._config_dict)

    @property
    def _state_spec(self) -> tuple[ArraySpec, ...]:
        return tuple(self._state_specs[key] for key in self._state_keys)

    @property
    def _action_spec(self) -> tuple[ArraySpec, ...]:
        return tuple(self._action_specs[key] for key in self._action_keys)

    @property
    def _state_keys(self) -> list[str]:
        return list(self._state_specs)

    @property
    def _action_keys(self) -> list[str]:
        return list(self._action_specs)

    @property
    def _config_values(self) -> tuple[Any, ...]:
        return self.__config_values

    @_config_values.setter
    def _config_values(self, config_values: tuple[Any, ...]) -> None:
        self.__config_values = config_values

    @property
    def state_array_spec(self) -> dict[str, ArraySpec]:
        """Specs of observations and infos in EnvPool's flat format."""
        return dict(self._state_specs)

    @property
    def action_array_spec(self) -> dict[str, ArraySpec]:
        """Specs of actions in EnvPool's flat format."""
        return dict(self._action_specs)

    def observation_spec(self) -> Any:
        """Convert Gymnasium-Robotics observation/info specs to dm_env."""
        spec = {
            key.replace("obs:", "").replace("info:", ""): dm_spec_transform(
                key.replace(":", ".").split(".")[-1],
                value,
                "obs",
            )
            for key, value in self._state_specs.items()
            if key.startswith(("obs", "info"))
        }
        return to_namedtuple(
            "State",
            _sanitize_dm_namedtuple_tree(to_nested_dict(spec)),
        )

    def action_spec(self) -> Any:
        """Convert Gymnasium-Robotics action specs to dm_env."""
        spec = {
            key.replace("action:", ""): dm_spec_transform(
                key.replace(":", ".").split(".")[-1],
                value,
                "act",
            )
            for key, value in self._action_specs.items()
        }
        if len(spec) == 1:
            return list(spec.values())[0]
        return to_namedtuple(
            "Action",
            _sanitize_dm_namedtuple_tree(to_nested_dict(spec)),
        )

    @property
    def observation_space(self) -> gymnasium.Space[Any]:
        """Gym observation space."""
        return self._gymnasium_observation_space

    @property
    def action_space(self) -> gymnasium.Space[Any]:
        """Gym action space."""
        return self._gymnasium_action_space

    @property
    def gymnasium_observation_space(self) -> gymnasium.Space[Any]:
        """Gymnasium observation space."""
        return self._gymnasium_observation_space

    @property
    def gymnasium_action_space(self) -> gymnasium.Space[Any]:
        """Gymnasium action space."""
        return self._gymnasium_action_space

    @property
    def reward_threshold(self) -> float | None:
        """Reward threshold copied from Gymnasium's registry."""
        return self.config.reward_threshold

    def __repr__(self) -> str:
        """Prettify debug info."""
        config_info = pprint.pformat(self.config)[6:]
        return f"{self.__class__.__name__}{config_info}"


class _GymnasiumRoboticsEnvPoolBase:
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        spec: GymnasiumRoboticsEnvSpec,
        thread_pool: ThreadPoolArg = None,
    ) -> None:
        """Create one upstream env instance per EnvPool slot."""
        del thread_pool
        self.spec = spec
        self._spec = spec
        self._config = dict(
            zip(spec._config_keys, spec._config_values, strict=False)
        )
        self._num_envs = int(self._config["num_envs"])
        self._batch_size = int(self._config["batch_size"])
        self._max_num_players = int(self._config["max_num_players"])
        if self._batch_size not in {0, self._num_envs}:
            raise ValueError(
                "Gymnasium-Robotics adapter currently supports sync mode only: "
                "set batch_size=0 or batch_size=num_envs."
            )
        if self._max_num_players != 1:
            raise ValueError(
                "Gymnasium-Robotics adapter only supports max_num_players=1."
            )

        base_seed = int(self._config["seed"])
        env_seed = self._config["env_seed"]
        if env_seed:
            self._seed = [_seed_for_env(int(seed)) for seed in env_seed]
        else:
            self._seed = [
                _seed_for_env(base_seed + env_id)
                for env_id in range(self._num_envs)
            ]
        self._needs_seeded_reset = [True] * self._num_envs
        self._elapsed_step = np.zeros(self._num_envs, dtype=np.int32)
        self._envs = [
            _make_upstream_env(
                self._config["gymnasium_task_id"],
                int(self._config["max_episode_steps"]),
                self._config["gymnasium_robotics_kwargs"],
            )
            for _ in range(self._num_envs)
        ]
        self._obs: list[Any] = [None] * self._num_envs
        self._info: list[dict[str, Any]] = [
            dict() for _ in range(self._num_envs)
        ]
        self._pending_step: tuple[Any, ...] | None = None
        self._pending_reset = False
        for env_id in range(self._num_envs):
            self._reset_one(env_id)
        self._needs_seeded_reset = [True] * self._num_envs

    def _reset_one(self, env_id: int) -> tuple[Any, dict[str, Any]]:
        seed = self._seed[env_id] if self._needs_seeded_reset[env_id] else None
        obs, info = self._envs[env_id].reset(seed=seed)
        self._needs_seeded_reset[env_id] = False
        self._obs[env_id] = obs
        self._info[env_id] = dict(info)
        self._elapsed_step[env_id] = 0
        return obs, self._info[env_id]

    def __len__(self) -> int:
        return self._num_envs

    @property
    def all_env_ids(self) -> np.ndarray:
        return np.arange(self._num_envs, dtype=np.int32)

    @property
    def is_async(self) -> bool:
        return False

    @property
    def observation_space(self) -> gymnasium.Space[Any]:
        return self.spec.gymnasium_observation_space

    @property
    def action_space(self) -> gymnasium.Space[Any]:
        return self.spec.gymnasium_action_space

    @property
    def render_mode(self) -> str | None:
        return getattr(self, "_render_mode", None)

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._config)

    def observation_spec(self) -> Any:
        return self.spec.observation_spec()

    def action_spec(self) -> Any:
        return self.spec.action_spec()

    def seed(self, seed: int | list[int] | None = None) -> None:
        del seed

    def _step_selected(
        self,
        action: Any,
        env_ids: np.ndarray,
    ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        info_list = []
        elapsed_step_list = []
        for batch_idx, env_id in enumerate(env_ids.tolist()):
            self._needs_seeded_reset[env_id] = False
            obs, reward, terminated, truncated, info = self._envs[env_id].step(
                _action_at(action, batch_idx, self.action_space)
            )
            elapsed_step = int(self._elapsed_step[env_id]) + 1
            obs_list.append(obs)
            reward_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            info_list.append(dict(info))
            elapsed_step_list.append(elapsed_step)
            if terminated or truncated:
                self._reset_one(env_id)
            else:
                self._obs[env_id] = obs
                self._info[env_id] = dict(info)
                self._elapsed_step[env_id] = elapsed_step
        return (
            _stack_space_values(obs_list, self.observation_space),
            np.asarray(reward_list),
            np.asarray(terminated_list, dtype=np.bool_),
            np.asarray(truncated_list, dtype=np.bool_),
            _stack_infos(
                info_list,
                env_ids,
                np.asarray(elapsed_step_list, dtype=np.int32),
            ),
        )

    def _reset_selected(
        self,
        env_ids: np.ndarray,
    ) -> tuple[Any, dict[str, np.ndarray]]:
        obs_list = []
        info_list = []
        elapsed_step = []
        for env_id in env_ids.tolist():
            obs, info = self._reset_one(env_id)
            obs_list.append(obs)
            info_list.append(info)
            elapsed_step.append(self._elapsed_step[env_id])
        return (
            _stack_space_values(obs_list, self.observation_space),
            _stack_infos(
                info_list,
                env_ids,
                np.asarray(elapsed_step, dtype=np.int32),
            ),
        )

    def send(
        self,
        action: Any,
        env_id: np.ndarray | None = None,
    ) -> None:
        env_ids = _as_int32_env_ids(env_id, self.all_env_ids)
        self._pending_step = self._step_selected(action, env_ids)
        self._pending_reset = False

    def recv(
        self,
        reset: bool = False,
        return_info: bool = True,
    ) -> Any:
        del reset, return_info
        if self._pending_step is None:
            raise RuntimeError("send() must be called before recv().")
        ret = self._pending_step
        self._pending_step = None
        self._pending_reset = False
        return ret

    def async_reset(self) -> None:
        obs, info = self._reset_selected(self.all_env_ids)
        num_envs = info["env_id"].shape[0]
        self._pending_step = (
            obs,
            np.zeros(num_envs, dtype=np.float64),
            np.zeros(num_envs, dtype=np.bool_),
            np.zeros(num_envs, dtype=np.bool_),
            info,
        )
        self._pending_reset = True

    def render(
        self,
        env_ids: int | Sequence[int] | np.ndarray | None = None,
        camera_id: int | None = None,
    ) -> np.ndarray | None:
        del camera_id
        render_mode = self.render_mode
        if render_mode not in {"rgb_array", "human"}:
            raise RuntimeError(
                "render_mode must be set to 'rgb_array' or 'human' when creating this env"
            )
        default_env_id = int(getattr(self, "_render_env_id", 0))
        selected_env_ids = _as_int32_env_ids(
            env_ids,
            np.asarray([default_env_id], dtype=np.int32),
        )
        frame_list = [
            cast(np.ndarray, self._envs[int(env_id)].render())
            for env_id in selected_env_ids
        ]
        frames: np.ndarray = np.stack(
            frame_list,
            axis=0,
        )
        if render_mode == "rgb_array":
            return frames
        if selected_env_ids.shape[0] != 1:
            raise ValueError(
                "render_mode='human' only supports a single env_id"
            )
        try_ensure_mujoco_glfw_context(frames.shape[2], frames.shape[1])
        import cv2

        cv2.imshow(
            f"{self.__class__.__name__}-render",
            np.ascontiguousarray(frames[0, :, :, ::-1]),
        )
        cv2.waitKey(1)
        return None

    def close(self) -> None:
        for env in self._envs:
            with contextlib.suppress(Exception):
                env.close()

    def xla(self) -> None:
        raise RuntimeError(
            "XLA is unavailable for the Gymnasium-Robotics Python adapter."
        )

    def __repr__(self) -> str:
        config_str = ", ".join([
            f"{k}={pprint.pformat(v)}" for k, v in self._config.items()
        ])
        return f"{self.__class__.__name__}({config_str})"

    def __str__(self) -> str:
        return self.__repr__()


class GymnasiumRoboticsGymnasiumEnvPool(  # type: ignore[misc,override,unused-ignore]
    _GymnasiumRoboticsEnvPoolBase,
    gymnasium.Env,
):
    """Gymnasium EnvPool adapter around upstream Gymnasium-Robotics envs."""

    def reset(  # type: ignore[override,unused-ignore]
        self,
        env_id: np.ndarray | None = None,
    ) -> tuple[Any, dict[str, np.ndarray]]:
        """Reset a batch of Gymnasium-Robotics envs."""
        env_ids = _as_int32_env_ids(env_id, self.all_env_ids)
        return self._reset_selected(env_ids)

    def step(  # type: ignore[override,unused-ignore]
        self,
        action: Any,
        env_id: np.ndarray | None = None,
    ) -> tuple[
        Any,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, np.ndarray],
    ]:
        """Step a batch of Gymnasium-Robotics envs."""
        env_ids = _as_int32_env_ids(env_id, self.all_env_ids)
        return self._step_selected(action, env_ids)


class GymnasiumRoboticsGymEnvPool(GymnasiumRoboticsGymnasiumEnvPool):
    """Gym-compatible alias of the Gymnasium adapter."""


class GymnasiumRoboticsDMEnvPool(
    _GymnasiumRoboticsEnvPoolBase,
    dm_env.Environment,
):
    """dm_env wrapper around upstream Gymnasium-Robotics envs."""

    def reset(self, env_id: np.ndarray | None = None) -> TimeStep:
        """Reset a batch of dm_env-compatible Gymnasium-Robotics envs."""
        env_ids = _as_int32_env_ids(env_id, self.all_env_ids)
        obs, info = self._reset_selected(env_ids)
        return TimeStep(
            step_type=np.full(
                env_ids.shape[0],
                dm_env.StepType.FIRST,
                dtype=np.int32,
            ),
            reward=np.zeros(env_ids.shape[0], dtype=np.float64),
            discount=np.ones(env_ids.shape[0], dtype=np.float64),
            observation=_build_dm_observation(obs, info),
        )

    def step(
        self,
        action: Any,
        env_id: np.ndarray | None = None,
    ) -> TimeStep:
        """Step a batch of dm_env-compatible Gymnasium-Robotics envs."""
        env_ids = _as_int32_env_ids(env_id, self.all_env_ids)
        return _dm_timestep(*self._step_selected(action, env_ids))

    def recv(
        self,
        reset: bool = False,
        return_info: bool = True,
    ) -> TimeStep:
        """Receive the pending async step as a dm_env timestep."""
        del reset, return_info
        if self._pending_step is None:
            raise RuntimeError("send() must be called before recv().")
        pending_step = self._pending_step
        pending_reset = self._pending_reset
        self._pending_step = None
        self._pending_reset = False
        if pending_reset:
            obs, _, _, _, info = pending_step
            return TimeStep(
                step_type=np.full(
                    info["env_id"].shape[0],
                    dm_env.StepType.FIRST,
                    dtype=np.int32,
                ),
                reward=np.zeros(info["env_id"].shape[0], dtype=np.float64),
                discount=np.ones(info["env_id"].shape[0], dtype=np.float64),
                observation=_build_dm_observation(obs, info),
            )
        return _dm_timestep(*pending_step)
