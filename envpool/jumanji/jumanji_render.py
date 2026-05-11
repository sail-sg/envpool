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
"""Python/Matplotlib rendering override for EnvPool's Jumanji tasks.

The actual Matplotlib viewers are adapted from Apache-2.0 Jumanji v1.1.1
under ``_official_render``. This module only caches EnvPool observations and
routes ``env.render()`` to those vendored viewers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from envpool.jumanji.jumanji_official_render import (
    configure_matplotlib,
    render_official_frame,
    update_render_aux,
)
from envpool.python.envpool import _normalize_render_env_ids

configure_matplotlib()

_GymEnvT = TypeVar("_GymEnvT", bound=type)


def _slice_tree(value: Any, index: int) -> Any:
    if isinstance(value, Mapping):
        return {key: _slice_tree(item, index) for key, item in value.items()}
    array = np.asarray(value)
    if array.shape[:1] == (0,):
        return array
    if array.ndim == 0:
        return array
    return np.array(array[index], copy=True)


def _slice_info(info: Mapping[str, Any], index: int) -> dict[str, Any]:
    sliced = {}
    for key, value in info.items():
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] > index:
            sliced[key] = np.array(array[index], copy=True)
        else:
            sliced[key] = value
    return sliced


def _empty_cache(env: Any) -> None:
    if not hasattr(env, "_jumanji_render_obs_cache"):
        env._jumanji_render_obs_cache = {}
        env._jumanji_render_info_cache = {}
        env._jumanji_render_score_cache = {}
        env._jumanji_render_aux_cache = {}


def _close_render_aux(aux: Any) -> None:
    if not isinstance(aux, Mapping):
        return
    for viewer in aux.get("_viewer_cache", {}).values():
        close = getattr(viewer, "close", None)
        if callable(close):
            close()


def _slice_action(action: Any, batch_index: int, batch_size: int) -> Any:
    if action is None:
        return None
    if isinstance(action, Mapping):
        return _slice_tree(action, batch_index)
    array = np.asarray(action)
    if array.ndim > 0 and array.shape[0] == batch_size:
        return np.array(array[batch_index], copy=True)
    return np.array(array, copy=True)


def _cache_gymnasium_output(
    env: Any,
    output: Any,
    reset: bool,
    action: Any = None,
) -> None:
    _empty_cache(env)
    if reset:
        obs, info = output
        reward = None
    else:
        obs, reward, _, _, info = output
    env_ids = np.asarray(info["env_id"], dtype=np.int32).reshape(-1)
    rewards = None if reward is None else np.asarray(reward).reshape(-1)
    batch_size = int(env_ids.shape[0])
    config = getattr(env, "config", {})
    for batch_index, env_id in enumerate(env_ids.tolist()):
        env_id_int = int(env_id)
        obs_slice = _slice_tree(obs, batch_index)
        info_slice = _slice_info(info, batch_index)
        previous_obs = env._jumanji_render_obs_cache.get(env_id_int)
        if reset:
            _close_render_aux(env._jumanji_render_aux_cache.get(env_id_int))
        action_slice = (
            None if reset else _slice_action(action, batch_index, batch_size)
        )
        env._jumanji_render_aux_cache[env_id_int] = update_render_aux(
            env._jumanji_task_id,
            env._jumanji_render_aux_cache.get(env_id_int),
            obs_slice,
            config,
            reset=reset,
            previous_obs=previous_obs,
            action=action_slice,
        )
        env._jumanji_render_obs_cache[env_id_int] = obs_slice
        env._jumanji_render_info_cache[env_id_int] = info_slice
        if reset:
            env._jumanji_render_score_cache[env_id_int] = 0.0
        elif rewards is not None:
            env._jumanji_render_score_cache[env_id_int] = float(
                env._jumanji_render_score_cache.get(env_id_int, 0.0)
            ) + float(rewards[batch_index])


def _jumanji_render(
    self: Any,
    env_ids: Any = None,
    camera_id: int | None = None,
) -> NDArray[np.uint8] | None:
    del camera_id
    render_mode, default_env_id, width, height, _ = self._render_config()
    width = 256 if width <= 0 else width
    height = 256 if height <= 0 else height
    if render_mode not in {"rgb_array", "human"}:
        raise RuntimeError(
            "render_mode must be set to 'rgb_array' or 'human' when creating this env"
        )
    env_ids_arr = _normalize_render_env_ids(env_ids, default_env_id)
    _empty_cache(self)
    frames = []
    config = getattr(self, "config", {})
    for env_id in env_ids_arr.tolist():
        if int(env_id) not in self._jumanji_render_obs_cache:
            raise RuntimeError(
                "Jumanji render requires reset() before render()."
            )
        obs = self._jumanji_render_obs_cache[int(env_id)]
        info = self._jumanji_render_info_cache.get(int(env_id), {})
        score = float(self._jumanji_render_score_cache.get(int(env_id), 0.0))
        aux = self._jumanji_render_aux_cache.get(int(env_id), {})
        frames.append(
            render_official_frame(
                self._jumanji_task_id,
                obs,
                info,
                config,
                aux,
                width,
                height,
                score,
            )
        )
    batch = np.stack(frames, axis=0).astype(np.uint8, copy=False)
    if render_mode == "human":
        if batch.shape[0] != 1:
            raise ValueError(
                "render_mode='human' only supports a single env_id"
            )
        self._show_human_frame(batch[0])
        return None
    return batch


def with_jumanji_python_render(cls: _GymEnvT, task_id: str) -> _GymEnvT:
    """Install a task-specific Python render override on a GymnasiumEnvPool class."""
    env_cls = cast(Any, cls)
    original_reset = cast(Callable[..., Any], env_cls.reset)
    original_step = cast(Callable[..., Any], env_cls.step)
    original_recv = cast(Callable[..., Any], env_cls.recv)
    original_close = getattr(env_cls, "close", None)

    def reset(self: Any, *args: Any, **kwargs: Any) -> Any:
        output = original_reset(self, *args, **kwargs)
        _cache_gymnasium_output(self, output, reset=True)
        return output

    def step(self: Any, *args: Any, **kwargs: Any) -> Any:
        action = args[0] if args else kwargs.get("action", None)
        output = original_step(self, *args, **kwargs)
        _cache_gymnasium_output(self, output, reset=False, action=action)
        return output

    def recv(self: Any, *args: Any, **kwargs: Any) -> Any:
        reset_output = (
            bool(args[0]) if args else bool(kwargs.get("reset", False))
        )
        output = original_recv(self, *args, **kwargs)
        _cache_gymnasium_output(self, output, reset=reset_output)
        return output

    def close(self: Any, *args: Any, **kwargs: Any) -> Any:
        for aux in getattr(self, "_jumanji_render_aux_cache", {}).values():
            _close_render_aux(aux)
        if callable(original_close):
            return original_close(self, *args, **kwargs)
        return None

    env_cls._jumanji_task_id = task_id
    env_cls.reset = reset
    env_cls.step = step
    env_cls.recv = recv
    env_cls.close = close
    env_cls.render = _jumanji_render
    return cls
