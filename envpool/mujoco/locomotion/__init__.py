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
"""Native Composer locomotion; Python only names and shapes native buffers."""

from collections import namedtuple
from functools import cached_property
from typing import Any

import dm_env
import gymnasium
import numpy as np

from envpool.python.api import py_env
from envpool.python.glfw_context import preload_windows_gl_dlls

preload_windows_gl_dlls(strict=True)

from .locomotion_envpool import (
    _LocomotionEnvPool,
    _LocomotionEnvSpec,
    _observation_layout,
)

LocomotionEnvSpec: Any
LocomotionDMEnvPool: Any
LocomotionGymnasiumEnvPool: Any
LocomotionEnvSpec, LocomotionDMEnvPool, LocomotionGymnasiumEnvPool = py_env(
    _LocomotionEnvSpec, _LocomotionEnvPool
)

_State = namedtuple("_State", ("obs", "env_id", "players"))
_Players = namedtuple("_Players", ("env_id",))
_DTYPES = (np.float64, np.int64, np.uint8)
_KEYS = dict(
    zip(
        _LocomotionEnvPool._state_keys,
        range(len(_LocomotionEnvPool._state_keys)),
        strict=True,
    )
)


def _layout(spec: Any) -> list[tuple]:
    return _observation_layout(spec.config.task_name, spec.config.team_size)


def _observation_spec(spec: Any) -> _State:
    obs = {
        name: dm_env.specs.Array(
            tuple(shape), np.bool_ if boolean else _DTYPES[storage], name=name
        )
        for name, shape, storage, boolean, _, _ in spec._layout
    }
    return _State(
        obs,
        dm_env.specs.Array((), np.int32),
        _Players(dm_env.specs.Array((), np.int32)),
    )


def _observation_space(spec: Any) -> gymnasium.spaces.Dict:
    spaces: dict[str, gymnasium.Space] = {}
    for name, shape, storage, boolean, _, _ in spec._layout:
        dtype: Any = np.bool_ if boolean else _DTYPES[storage]
        low, high = (
            (0, 1)
            if boolean
            else (0, 255)
            if storage == 2
            else (-np.inf, np.inf)
            if storage == 0
            else (np.iinfo(np.int64).min, np.iinfo(np.int64).max)
        )
        spaces[name] = gymnasium.spaces.Box(
            low, high, tuple(shape), dtype=dtype
        )
    return gymnasium.spaces.Dict(spaces)


def _unpack(
    self: Any, values: list[np.ndarray]
) -> tuple[dict, dict, np.ndarray]:
    buffers = [
        values[_KEYS[f"obs:{key}"]]
        for key in ("continuous", "discrete", "pixels")
    ]
    obs = {}
    for name, shape, storage, boolean, offset, size in self.spec._layout:
        buffer = buffers[storage]
        value = buffer[:, offset : offset + size].reshape((len(buffer), *shape))
        obs[name] = value.astype(np.bool_) if boolean else value
    info = {
        "env_id": values[_KEYS["info:env_id"]],
        "players": {"env_id": values[_KEYS["info:players.env_id"]]},
        "elapsed_step": values[_KEYS["elapsed_step"]],
    }
    # Composer single-player rewards are float64. Soccer explicitly returns
    # float32 per player, which is already the common EnvPool reward dtype.
    reward = values[
        _KEYS[
            "reward"
            if self.config["task_name"].startswith("soccer_")
            else "reward64"
        ]
    ]
    return obs, info, reward


def _to_dm(
    self: Any, values: list[np.ndarray], reset: bool, return_info: bool
) -> dm_env.TimeStep:
    obs, info, reward = _unpack(self, values)
    return dm_env.TimeStep(
        values[_KEYS["step_type"]],
        reward,
        values[_KEYS["discount"]].astype(reward.dtype, copy=False),
        _State(obs, info["env_id"], _Players(info["players"]["env_id"])),
    )


def _reward_spec(self: Any) -> dm_env.specs.Array:
    dtype = (
        np.float32
        if self.config["task_name"].startswith("soccer_")
        else np.float64
    )
    return dm_env.specs.Array((), dtype, name="reward")


def _discount_spec(self: Any) -> dm_env.specs.BoundedArray:
    return dm_env.specs.BoundedArray(
        (), _reward_spec(self).dtype, 0, 1, name="discount"
    )


def _to_gymnasium(
    self: Any, values: list[np.ndarray], reset: bool, return_info: bool
) -> tuple:
    obs, info, reward = _unpack(self, values)
    if reset:
        return obs, info
    return (
        obs,
        reward,
        values[_KEYS["terminated"]],
        values[_KEYS["trunc"]],
        info,
    )


# Keep the standard batching, action validation, autoreset and rendering APIs.
# Only dictionary layout and reward precision differ from py_env's defaults.
LocomotionEnvSpec._layout = cached_property(_layout)
LocomotionEnvSpec._layout.__set_name__(LocomotionEnvSpec, "_layout")
LocomotionEnvSpec.observation_spec = _observation_spec
LocomotionEnvSpec.observation_space = property(_observation_space)
LocomotionDMEnvPool._to = _to_dm
LocomotionDMEnvPool.reward_spec = _reward_spec
LocomotionDMEnvPool.discount_spec = _discount_spec
LocomotionGymnasiumEnvPool._to = _to_gymnasium
