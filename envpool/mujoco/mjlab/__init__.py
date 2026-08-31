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
"""Native MJLab tasks; Python only names and reshapes native buffers."""

import json
from collections import namedtuple
from functools import cached_property
from typing import Any

import dm_env
import gymnasium
import numpy as np

from envpool.python.api import py_env
from envpool.python.glfw_context import preload_windows_gl_dlls

preload_windows_gl_dlls(strict=True)

from .mjlab_envpool import REGISTRY_JSON, _MjlabEnvPool, _MjlabEnvSpec

TASKS = {task["id"]: task for task in json.loads(REGISTRY_JSON)}

MjlabEnvSpec: Any
MjlabDMEnvPool: Any
MjlabGymnasiumEnvPool: Any
MjlabEnvSpec, MjlabDMEnvPool, MjlabGymnasiumEnvPool = py_env(
    _MjlabEnvSpec, _MjlabEnvPool
)

_State = namedtuple("_State", ("obs", "env_id", "players"))
_Players = namedtuple("_Players", ("env_id",))
_KEYS = {key: i for i, key in enumerate(_MjlabEnvPool._state_keys)}


def _layout(spec: Any) -> list[tuple[str, tuple[int, ...], int, int]]:
    offset = 0
    layout = []
    for name, shape in sorted(
        TASKS[spec.config.task_name]["observation_shapes"].items()
    ):
        size = int(np.prod(shape))
        layout.append((name, tuple(shape), offset, size))
        offset += size
    return layout


def _observation_spec(spec: Any) -> _State:
    return _State(
        {
            name: dm_env.specs.Array(shape, np.float32, name=name)
            for name, shape, _, _ in spec._layout
        },
        dm_env.specs.Array((), np.int32),
        _Players(dm_env.specs.Array((), np.int32)),
    )


def _observation_space(spec: Any) -> gymnasium.spaces.Dict:
    return gymnasium.spaces.Dict({
        name: gymnasium.spaces.Box(-np.inf, np.inf, shape, np.float32)
        for name, shape, _, _ in spec._layout
    })


def _unpack(self: Any, values: list[np.ndarray]) -> tuple[dict, dict]:
    buffer = values[_KEYS["obs"]]
    obs = {
        name: buffer[:, offset : offset + size].reshape((len(buffer), *shape))
        for name, shape, offset, size in self.spec._layout
    }
    info = {
        "env_id": values[_KEYS["info:env_id"]],
        "players": {"env_id": values[_KEYS["info:players.env_id"]]},
        "elapsed_step": values[_KEYS["elapsed_step"]],
    }
    return obs, info


def _to_dm(
    self: Any, values: list[np.ndarray], reset: bool, return_info: bool
) -> dm_env.TimeStep:
    obs, info = _unpack(self, values)
    return dm_env.TimeStep(
        values[_KEYS["step_type"]],
        values[_KEYS["reward"]],
        values[_KEYS["discount"]],
        _State(obs, info["env_id"], _Players(info["players"]["env_id"])),
    )


def _to_gymnasium(
    self: Any, values: list[np.ndarray], reset: bool, return_info: bool
) -> tuple:
    obs, info = _unpack(self, values)
    if reset:
        return obs, info
    return (
        obs,
        values[_KEYS["reward"]],
        values[_KEYS["terminated"]],
        values[_KEYS["trunc"]],
        info,
    )


MjlabEnvSpec._layout = cached_property(_layout)
MjlabEnvSpec._layout.__set_name__(MjlabEnvSpec, "_layout")
MjlabEnvSpec.observation_spec = _observation_spec
MjlabEnvSpec.observation_space = property(_observation_space)
MjlabDMEnvPool._to = _to_dm
MjlabGymnasiumEnvPool._to = _to_gymnasium
