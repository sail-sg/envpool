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
"""EnvPool meta class for gymnasium.Env API."""

import warnings
from abc import ABCMeta
from typing import Any, cast

import gymnasium
import numpy as np
import optree

from .data import gymnasium_structure
from .envpool import EnvPoolMixin
from .utils import check_key_duplication

try:
    from gymnasium.vector import VectorEnv as _GymnasiumVectorEnv
except (AttributeError, ImportError):
    _VECTOR_ENV_CLS: type | None = None
else:
    _VECTOR_ENV_CLS = _GymnasiumVectorEnv

try:
    from gymnasium.vector.vector_env import AutoresetMode as _AutoresetMode
except (AttributeError, ImportError):
    _AUTORESET_MODE = None
else:
    _AUTORESET_MODE = _AutoresetMode.NEXT_STEP


def _gymnasium_base_classes() -> tuple[type, ...]:
    if _VECTOR_ENV_CLS is None:
        return (gymnasium.Env,)
    if issubclass(_VECTOR_ENV_CLS, gymnasium.Env):
        return (_VECTOR_ENV_CLS,)
    return (_VECTOR_ENV_CLS, gymnasium.Env)


def _env_ids_from_reset_options(
    options: dict[str, Any] | None, num_envs: int
) -> np.ndarray | None:
    if options is None:
        return None
    allowed_options = {"reset_mask"}
    unknown_options = set(options) - allowed_options
    if unknown_options:
        raise ValueError(
            "Unsupported Gymnasium reset options for EnvPool: "
            f"{sorted(unknown_options)}"
        )
    reset_mask = options.get("reset_mask")
    if reset_mask is None:
        return None
    reset_mask = np.asarray(reset_mask, dtype=np.bool_)
    if reset_mask.shape != (num_envs,):
        raise ValueError(
            f"reset_mask must have shape ({num_envs},), got {reset_mask.shape}"
        )
    if not np.any(reset_mask):
        raise ValueError("reset_mask must select at least one environment.")
    return np.flatnonzero(reset_mask).astype(np.int32)


class GymnasiumEnvPoolMixin:
    """Special treatment for gymnasim API."""

    metadata = (
        {
            "render_modes": ["rgb_array", "human"],
            "autoreset_mode": _AUTORESET_MODE,
        }
        if _AUTORESET_MODE is not None
        else {"render_modes": ["rgb_array", "human"]}
    )

    @property
    def num_envs(self: Any) -> int:
        """Number of sub-environments in this vectorized EnvPool."""
        return int(self.config["num_envs"])

    @property
    def is_vector_env(self: Any) -> bool:
        """Compatibility flag used by older Gymnasium vector-aware wrappers."""
        return True

    @property
    def single_observation_space(
        self: Any,
    ) -> gymnasium.Space | dict[str, Any]:
        """Single sub-environment observation space."""
        return self.observation_space

    @property
    def single_action_space(self: Any) -> gymnasium.Space | dict[str, Any]:
        """Single sub-environment action space."""
        return self.action_space

    @property
    def observation_space(self: Any) -> gymnasium.Space | dict[str, Any]:
        """Observation space from EnvSpec."""
        if not hasattr(self, "_gym_observation_space"):
            self._gym_observation_space = self.spec.gymnasium_observation_space
        return self._gym_observation_space

    @property
    def action_space(self: Any) -> gymnasium.Space | dict[str, Any]:
        """Action space from EnvSpec."""
        if not hasattr(self, "_gym_action_space"):
            self._gym_action_space = self.spec.gymnasium_action_space
        return self._gym_action_space

    @property
    def render_mode(self: Any) -> str | None:
        """Render mode configured at construction time."""
        return getattr(self, "_render_mode", None)

    def reset(
        self: Any,
        env_id: np.ndarray | None = None,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> Any:
        """Reset with Gymnasium-compatible seed and options keywords."""
        if seed is not None:
            warnings.warn(
                "EnvPool seeds are fixed when the environment is created. "
                "reset(seed=...) is ignored; pass seed to envpool.make "
                "instead.",
                stacklevel=2,
            )
        option_env_id = _env_ids_from_reset_options(
            options, self.config["num_envs"]
        )
        if env_id is not None and option_env_id is not None:
            raise ValueError(
                "Pass either env_id or options['reset_mask'], not both."
            )
        if option_env_id is not None:
            env_id = option_env_id
        return cast(Any, super()).reset(env_id)

    def close(self: Any, **kwargs: Any) -> None:
        """Accept Gymnasium VectorEnv close kwargs without changing EnvPool."""
        del kwargs
        return cast(Any, super()).close()


class GymnasiumEnvPoolMeta(
    ABCMeta,
    gymnasium.Env.__class__,  # type: ignore[valid-type,misc,unused-ignore]
):
    """Additional wrapper for EnvPool gymnasium.Env API."""

    def __new__(cls: Any, name: str, parents: tuple, attrs: dict) -> Any:
        """Check internal config and initialize data format convertion."""
        base = parents[0]
        try:
            from .lax import XlaMixin

            parents = (
                base,
                GymnasiumEnvPoolMixin,
                EnvPoolMixin,
                XlaMixin,
                *_gymnasium_base_classes(),
            )
        except (ImportError, AttributeError):

            def _xla(self: Any) -> None:
                raise RuntimeError(
                    "XLA is unavailable. To enable XLA please install a compatible jax."
                )

            attrs["xla"] = _xla
            parents = (
                base,
                GymnasiumEnvPoolMixin,
                EnvPoolMixin,
                *_gymnasium_base_classes(),
            )

        state_keys = base._state_keys
        action_keys = base._action_keys
        check_key_duplication(name, "state", state_keys)
        check_key_duplication(name, "action", action_keys)

        state_paths, state_idx, treepsec = gymnasium_structure(state_keys)

        def _to_gymnasium(
            self: Any,
            state_values: list[np.ndarray],
            reset: bool,
            return_info: bool,
        ) -> (
            Any
            | tuple[Any, Any]
            | tuple[Any, np.ndarray, np.ndarray, Any]
            | tuple[Any, np.ndarray, np.ndarray, np.ndarray, Any]
        ):
            values = (state_values[i] for i in state_idx)
            state = cast(
                dict[str, Any], optree.tree_unflatten(treepsec, values)
            )
            info = cast(dict[str, Any], state["info"])
            info["elapsed_step"] = state["elapsed_step"]
            obs = state["obs"]
            if not isinstance(self.observation_space, gymnasium.spaces.Dict):
                while isinstance(obs, dict) and len(obs) == 1:
                    obs = next(iter(obs.values()))
            if reset:
                return obs, info
            done = cast(np.ndarray, state["done"])
            trunc = cast(np.ndarray, state["trunc"])
            terminated = done & ~trunc
            return obs, state["reward"], terminated, trunc, info

        attrs["_to"] = _to_gymnasium
        subcls = super().__new__(cls, name, parents, attrs)

        def init(self: Any, spec: Any) -> None:
            """Set self.spec to EnvSpecMeta."""
            super(subcls, self).__init__(spec)
            self.spec = spec

        setattr(subcls, "__init__", init)  # noqa: B010
        return subcls
