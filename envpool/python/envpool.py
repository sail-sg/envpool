# Copyright 2021 Garena Online Private Limited
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
"""EnvPool Mixin class for meta class definition."""

import pprint
import sys
import warnings
from abc import ABC
from typing import Any, cast

import numpy as np
import optree
from dm_env import TimeStep

from .glfw_context import ensure_mujoco_glfw_context
from .protocol import EnvPool, EnvSpec


def _normalize_env_id(env_id: Any) -> Any:
    """Normalize env_id while preserving traced arrays for XLA send paths."""
    if isinstance(env_id, np.ndarray):
        env_id = env_id.astype(np.int32, copy=False)
    elif hasattr(env_id, "astype"):
        env_id = env_id.astype(np.int32)
    else:
        env_id = np.asarray(env_id, dtype=np.int32)
    if getattr(env_id, "ndim", 0) == 0:
        env_id = env_id.reshape(1)
    return env_id


def _normalize_render_env_ids(env_ids: Any, default_env_id: int) -> np.ndarray:
    if env_ids is None:
        env_ids = np.asarray([default_env_id], dtype=np.int32)
    elif isinstance(env_ids, (int, np.integer)):
        env_ids = np.asarray([env_ids], dtype=np.int32)
    else:
        env_ids = _normalize_env_id(env_ids)
    return np.asarray(env_ids, dtype=np.int32)


class EnvPoolMixin(ABC):
    """Mixin class for EnvPool, exposed to EnvPoolMeta."""

    _spec: EnvSpec

    def _ensure_platform_render_context(
        self: EnvPool, width: int, height: int
    ) -> None:
        if (
            sys.platform == "win32"
            and self.__class__.__module__.startswith("envpool.mujoco")
        ):
            ensure_mujoco_glfw_context(width or 640, height or 480)

    def _player_action_count(
        self: EnvPool, adict: dict[str, Any]
    ) -> int | None:
        """Infer how many player actions are present in the current input."""
        player_count = None
        for key, spec in self.spec.action_array_spec.items():
            if key in ("env_id", "players.env_id") or key not in adict:
                continue
            shape = tuple(spec.shape)
            if len(shape) == 0 or shape[0] != -1:
                continue
            value_shape = np.shape(adict[key])
            count = 1 if len(value_shape) == 0 else int(value_shape[0])
            if player_count is None:
                player_count = count
            elif player_count != count:
                raise RuntimeError(
                    "Inconsistent leading dimensions across player actions."
                )
        return player_count

    def _cached_players_env_id(
        self: EnvPool, env_id: np.ndarray, player_count: int
    ) -> np.ndarray | None:
        """Reuse the last recv/reset mapping when player counts vary by env."""
        if not hasattr(self, "_last_players_env_id"):
            return None
        cached = np.asarray(self._last_players_env_id, dtype=np.int32)
        segments = []
        for eid in env_id.tolist():
            matches = cached[cached == eid]
            if matches.size == 0:
                return None
            segments.append(matches)
        if not segments:
            return np.empty(0, dtype=np.int32)
        players_env_id = np.concatenate(segments)
        if players_env_id.shape[0] != player_count:
            return None
        return players_env_id

    def _infer_players_env_id(
        self: EnvPool, adict: dict[str, Any]
    ) -> np.ndarray:
        """Fill in players.env_id for the simplified multiplayer API."""
        env_id = _normalize_env_id(adict["env_id"])
        if self.config.get("max_num_players", 1) == 1:
            return env_id
        env_id = np.asarray(env_id, dtype=np.int32)
        player_count = self._player_action_count(adict)
        if player_count is None or player_count == env_id.shape[0]:
            return env_id
        cached = self._cached_players_env_id(env_id, player_count)
        if cached is not None:
            return cached
        if env_id.shape[0] == 0 or player_count % env_id.shape[0] != 0:
            raise RuntimeError(
                "Cannot infer players.env_id for multiplayer action; "
                "pass a dict action with explicit players.env_id."
            )
        players_per_env = player_count // env_id.shape[0]
        max_num_players = self.config.get("max_num_players", 1)
        if players_per_env > max_num_players:
            raise RuntimeError(
                "Cannot infer players.env_id for multiplayer action; "
                "per-env player count exceeds max_num_players."
            )
        return np.repeat(env_id, players_per_env).astype(np.int32, copy=False)

    def _check_action(self: EnvPool, actions: list[np.ndarray]) -> None:
        if hasattr(self, "_check_action_finished"):  # only check once
            return
        self._check_action_finished = True
        for a, (k, v) in zip(
            actions, self.spec.action_array_spec.items(), strict=False
        ):
            if v.dtype != a.dtype:
                raise RuntimeError(
                    f'Expected dtype {v.dtype} with action "{k}", got {a.dtype}'
                )
            shape = tuple(v.shape)
            if len(shape) > 0 and shape[0] == -1:
                if a.shape[1:] != shape[1:]:
                    raise RuntimeError(
                        f'Expected shape {shape} with action "{k}", got {a.shape}'
                    )
            else:
                if len(a.shape) == 0 or a.shape[1:] != shape:
                    raise RuntimeError(
                        f'Expected shape {("num_env", *shape)} with action "{k}", got {a.shape}'
                    )

    def _from(
        self: EnvPool,
        action: dict[str, Any] | np.ndarray,
        env_id: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Convert action to C++-acceptable format."""
        if isinstance(action, dict):
            paths: list[tuple[str, ...]]
            values: list[Any]
            paths, values, _ = optree.tree_flatten_with_path(cast(Any, action))
            adict = {
                ".".join(p): v for p, v in zip(paths, values, strict=False)
            }
        else:  # only 3 keys in action_keys
            if not hasattr(self, "_last_action_type"):
                self._last_action_type = self._spec._action_spec[-1][0]
            if not hasattr(self, "_last_action_name"):
                self._last_action_name = self._spec._action_keys[-1]
            if isinstance(action, np.ndarray):
                # else it could be a jax array, when using xla
                action = action.astype(
                    self._last_action_type,
                    order="C",
                )
            adict = {self._last_action_name: action}
        if env_id is None:
            if "env_id" not in adict:
                adict["env_id"] = self.all_env_ids
        else:
            adict["env_id"] = env_id.astype(np.int32)
        if "players.env_id" not in adict:
            adict["players.env_id"] = self._infer_players_env_id(adict)
        if not hasattr(self, "_action_names"):
            self._action_names = self._spec._action_keys
        return [adict[k] for k in self._action_names]

    def __len__(self: EnvPool) -> int:
        """Return the number of environments."""
        return self.config["num_envs"]

    @property
    def all_env_ids(self: EnvPool) -> np.ndarray:
        """All env_id in numpy ndarray with dtype=np.int32."""
        if not hasattr(self, "_all_env_ids"):
            self._all_env_ids = np.arange(
                self.config["num_envs"], dtype=np.int32
            )
        return self._all_env_ids

    @property
    def is_async(self: EnvPool) -> bool:
        """Return if this env is in sync mode or async mode."""
        return (
            self.config["batch_size"] > 0
            and self.config["num_envs"] != self.config["batch_size"]
        )

    def seed(self: EnvPool, seed: int | list[int] | None = None) -> None:
        """Set the seed for all environments (abandoned)."""
        warnings.warn(
            "The `seed` function in envpool is abandoned. "
            "You can set seed by envpool.make(..., seed=seed) instead.",
            stacklevel=2,
        )

    def _render_config(self: EnvPool) -> tuple[str | None, int, int, int, int]:
        return (
            cast(str | None, getattr(self, "_render_mode", None)),
            int(getattr(self, "_render_env_id", 0)),
            int(getattr(self, "_render_width", 0)),
            int(getattr(self, "_render_height", 0)),
            int(getattr(self, "_render_camera_id", -1)),
        )

    def _show_human_frame(self: EnvPool, frame: np.ndarray) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "render_mode='human' requires opencv-python to be installed"
            ) from exc

        window_name = getattr(
            self, "_render_window_name", f"{self.__class__.__name__}-render"
        )
        cv2.imshow(window_name, np.ascontiguousarray(frame[:, :, ::-1]))
        cv2.waitKey(1)
        self._render_window_name = window_name
        self._render_window_open = True

    def render(
        self: EnvPool,
        env_ids: Any = None,
        camera_id: int | None = None,
    ) -> np.ndarray | None:
        """Render one or more environments using the configured render mode."""
        (
            render_mode,
            default_env_id,
            default_width,
            default_height,
            default_cam,
        ) = self._render_config()
        if render_mode not in {"rgb_array", "human"}:
            raise RuntimeError(
                "render_mode must be set to 'rgb_array' or 'human' when creating this env"
            )

        env_ids_arr = _normalize_render_env_ids(env_ids, default_env_id)
        width = default_width
        height = default_height
        camera_id = default_cam if camera_id is None else int(camera_id)
        self._ensure_platform_render_context(width, height)
        frames = self._render(env_ids_arr, width, height, camera_id)
        if render_mode == "human":
            if env_ids_arr.shape[0] != 1:
                raise ValueError(
                    "render_mode='human' only supports a single env_id"
                )
            self._show_human_frame(frames[0])
            return None
        return frames

    def send(
        self: EnvPool,
        action: dict[str, Any] | np.ndarray,
        env_id: np.ndarray | None = None,
    ) -> None:
        """Send actions into EnvPool."""
        converted_action = self._from(action, env_id)
        self._check_action(converted_action)
        self._send(converted_action)

    def recv(
        self: EnvPool,
        reset: bool = False,
        return_info: bool = True,
    ) -> TimeStep | tuple:
        """Recv a batch state from EnvPool."""
        state_list = self._recv()
        if not hasattr(self, "_state_names"):
            self._state_names = self._state_keys
        state = dict(zip(self._state_names, state_list, strict=False))
        if "info:players.env_id" in state:
            self._last_players_env_id = np.array(
                state["info:players.env_id"], copy=True
            )
        return self._to(state_list, reset, return_info)

    def async_reset(self: EnvPool) -> None:
        """Follows the async semantics, reset the envs in env_ids."""
        self._reset(self.all_env_ids)

    def step(
        self: EnvPool,
        action: dict[str, Any] | np.ndarray,
        env_id: np.ndarray | None = None,
    ) -> TimeStep | tuple:
        """Perform one step with multiple environments in EnvPool."""
        self.send(action, env_id)
        return self.recv(reset=False, return_info=True)

    def reset(
        self: EnvPool,
        env_id: np.ndarray | None = None,
    ) -> TimeStep | tuple:
        """Reset envs in env_id.

        This behavior is not defined in async mode.
        """
        if env_id is None:
            env_id = self.all_env_ids
        self._reset(env_id)
        return self.recv(
            reset=True, return_info=self.config["gym_reset_return_info"]
        )

    def close(self: EnvPool) -> None:
        """Close viewer resources and delegate to the parent implementation."""
        if getattr(self, "_render_window_open", False):
            try:
                import cv2

                cv2.destroyWindow(self._render_window_name)
            except Exception:
                pass
            self._render_window_open = False
        close = getattr(super(), "close", None)
        if close is not None:
            close()

    @property
    def config(self: EnvPool) -> dict[str, Any]:
        """Config dict of this class."""
        return dict(
            zip(
                self._spec._config_keys, self._spec._config_values, strict=False
            )
        )

    def __repr__(self: EnvPool) -> str:
        """Prettify the debug information."""
        config = self.config
        config_str = ", ".join([
            f"{k}={pprint.pformat(v)}" for k, v in config.items()
        ])
        return f"{self.__class__.__name__}({config_str})"

    def __str__(self: EnvPool) -> str:
        """Prettify the debug information."""
        return self.__repr__()
