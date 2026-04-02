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
"""EnvPool meta class for gym.Env API."""

from abc import ABCMeta
from typing import Any, cast

import gymnasium as gym
import numpy as np
import optree

from .data import gym_structure
from .envpool import EnvPoolMixin
from .protocol import (
    EnvPoolInit,
    EnvPoolInitNoThreadPool,
    EnvSpec,
    ThreadPoolArg,
)
from .utils import check_key_duplication


class GymEnvPoolMixin:
    """Special treatment for gym API."""

    metadata = {"render_modes": ["rgb_array", "human"]}

    @property
    def observation_space(self: Any) -> gym.Space | dict[str, Any]:
        """Observation space from EnvSpec."""
        if not hasattr(self, "_gym_observation_space"):
            self._gym_observation_space = self.spec.observation_space
        return self._gym_observation_space

    @property
    def action_space(self: Any) -> gym.Space | dict[str, Any]:
        """Action space from EnvSpec."""
        if not hasattr(self, "_gym_action_space"):
            self._gym_action_space = self.spec.action_space
        return self._gym_action_space

    @property
    def render_mode(self: Any) -> str | None:
        """Render mode configured at construction time."""
        return getattr(self, "_render_mode", None)


class GymEnvPoolMeta(
    ABCMeta,
    gym.Env.__class__,  # type: ignore[valid-type,misc,unused-ignore]
):
    """Additional wrapper for EnvPool gym.Env API."""

    def __new__(cls: Any, name: str, parents: tuple, attrs: dict) -> Any:
        """Check internal config and initialize data format convertion."""
        base = parents[0]
        try:
            from .lax import XlaMixin

            parents = (base, GymEnvPoolMixin, EnvPoolMixin, XlaMixin, gym.Env)
        except (ImportError, AttributeError):

            def _xla(self: Any) -> None:
                raise RuntimeError(
                    "XLA is unavailable. To enable XLA please install a compatible jax."
                )

            attrs["xla"] = _xla
            parents = (base, GymEnvPoolMixin, EnvPoolMixin, gym.Env)

        state_keys = base._state_keys
        action_keys = base._action_keys
        check_key_duplication(name, "state", state_keys)
        check_key_duplication(name, "action", action_keys)

        state_paths, state_idx, treepsec = gym_structure(state_keys)

        def _to_gym(
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
            if reset and not return_info:
                return state["obs"]
            info = cast(dict[str, Any], state["info"])
            info["elapsed_step"] = state["elapsed_step"]
            if reset:
                return state["obs"], info
            done = cast(np.ndarray, state["done"])
            trunc = cast(np.ndarray, state["trunc"])
            terminated = done & ~trunc
            return state["obs"], state["reward"], terminated, trunc, info

        attrs["_to"] = _to_gym
        subcls = super().__new__(cls, name, parents, attrs)

        def init(
            self: Any, spec: EnvSpec, thread_pool: ThreadPoolArg = None
        ) -> None:
            """Run the bound C++ constructor and cache the Python spec."""
            if thread_pool is None:
                cast(EnvPoolInitNoThreadPool, super(subcls, self).__init__)(
                    spec
                )
            else:
                cast(EnvPoolInit, super(subcls, self).__init__)(
                    spec, thread_pool
                )
            self.spec = spec

        setattr(subcls, "__init__", init)  # noqa: B010
        return subcls
