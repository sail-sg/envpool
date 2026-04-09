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
"""Global env registry."""

import importlib
import os
from collections.abc import Sequence
from typing import Any, Literal, overload

import numpy as np

from .python.protocol import (
    DMEnvPool,
    EnvSpec,
    GymnasiumEnvPool,
)

base_path = os.path.abspath(os.path.dirname(__file__))


class EnvRegistry:
    """A collection of available envs."""

    def __init__(self) -> None:
        """Constructor of EnvRegistry."""
        self.specs: dict[str, tuple[str, str, dict[str, Any]]] = {}
        self.envpools: dict[str, dict[str, tuple[str, str]]] = {}

    def register(
        self,
        task_id: str,
        import_path: str,
        spec_cls: str,
        dm_cls: str,
        gymnasium_cls: str,
        aliases: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        """Register EnvSpec and EnvPool in global EnvRegistry."""
        if "base_path" not in kwargs:
            kwargs["base_path"] = base_path
        for alias in (task_id, *aliases):
            assert alias not in self.specs
            self.specs[alias] = (import_path, spec_cls, dict(kwargs))
            self.envpools[alias] = {
                "dm": (import_path, dm_cls),
                "gymnasium": (import_path, gymnasium_cls),
            }

    @staticmethod
    def _extract_make_options(
        kwargs: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        from_pixels = bool(kwargs.pop("from_pixels", False))
        wrapper_kwargs = {
            key: kwargs.pop(key)
            for key in ("render_mode", "render_env_id")
            if key in kwargs
        }
        render_mode = wrapper_kwargs.get("render_mode")
        if render_mode not in {None, "rgb_array", "human"}:
            raise ValueError(
                "render_mode must be one of None, 'rgb_array', or 'human'"
            )
        if from_pixels:
            kwargs.setdefault("render_width", 84)
            kwargs.setdefault("render_height", 84)
            if (
                int(kwargs["render_width"]) <= 0
                or int(kwargs["render_height"]) <= 0
            ):
                raise ValueError(
                    "from_pixels=True requires positive render_width and "
                    "render_height"
                )
            for key in ("render_width", "render_height", "render_camera_id"):
                if key in kwargs and render_mode is not None:
                    wrapper_kwargs[key] = kwargs[key]
        else:
            for key in ("render_width", "render_height", "render_camera_id"):
                if key in kwargs:
                    wrapper_kwargs[key] = kwargs.pop(key)
        return from_pixels, wrapper_kwargs

    @staticmethod
    def _apply_wrapper_kwargs(env: Any, wrapper_kwargs: dict[str, Any]) -> Any:
        for key, value in wrapper_kwargs.items():
            setattr(env, f"_{key}", value)
        return env

    @staticmethod
    def _pixel_variant_name(class_name: str, suffix: str) -> str:
        if not class_name.endswith(suffix):
            raise ValueError(f"{class_name} does not end with {suffix}")
        return f"{class_name[: -len(suffix)]}Pixel{suffix}"

    @staticmethod
    def _pixel_variant_supported(import_path: str) -> bool:
        return import_path in {
            "envpool.mujoco.dmc",
            "envpool.mujoco.gym",
            "envpool.mujoco.robotics",
        }

    def _resolve_spec_entry(
        self, task_id: str, from_pixels: bool
    ) -> tuple[str, str, dict[str, Any]]:
        import_path, spec_cls, kwargs = self.specs[task_id]
        if from_pixels:
            if not self._pixel_variant_supported(import_path):
                raise ValueError(
                    "from_pixels=True is only supported for MuJoCo tasks."
                )
            spec_cls = self._pixel_variant_name(spec_cls, "EnvSpec")
        return import_path, spec_cls, dict(kwargs)

    def _resolve_envpool_entry(
        self, task_id: str, env_type: str, from_pixels: bool
    ) -> tuple[str, str]:
        import_path, envpool_cls = self.envpools[task_id][env_type]
        if from_pixels:
            if not self._pixel_variant_supported(import_path):
                raise ValueError(
                    "from_pixels=True is only supported for MuJoCo tasks."
                )
            suffix = {
                "dm": "DMEnvPool",
                "gymnasium": "GymnasiumEnvPool",
            }[env_type]
            envpool_cls = self._pixel_variant_name(envpool_cls, suffix)
        return import_path, envpool_cls

    def _make_env_spec(
        self, task_id: str, *, from_pixels: bool = False, **make_kwargs: Any
    ) -> EnvSpec:
        """Make the underlying EnvSpec used by EnvPool constructors."""
        import_path, spec_cls, kwargs = self._resolve_spec_entry(
            task_id, from_pixels
        )
        kwargs = {**kwargs, **make_kwargs}

        # check arguments
        if "seed" in kwargs:  # Issue 214
            if self._is_env_seed_sequence(kwargs["seed"]):
                assert "env_seed" not in kwargs, (
                    "Pass either `seed` as an int or seed list, or "
                    "`env_seed`, but not both."
                )
                kwargs["env_seed"] = self._normalize_env_seed(
                    kwargs["seed"],
                    kwargs.get("num_envs", 1),
                )
                kwargs["seed"] = 0
            else:
                self._assert_int32_seed(kwargs["seed"])
        if "env_seed" in kwargs:
            kwargs["env_seed"] = self._normalize_env_seed(
                kwargs["env_seed"],
                kwargs.get("num_envs", 1),
            )
        if "num_envs" in kwargs:
            assert kwargs["num_envs"] >= 1
        if "batch_size" in kwargs:
            assert 0 <= kwargs["batch_size"] <= kwargs["num_envs"]
        if "max_num_players" in kwargs:
            assert 1 <= kwargs["max_num_players"]

        spec_cls = getattr(importlib.import_module(import_path), spec_cls)
        config = spec_cls.gen_config(**kwargs)
        return spec_cls(config)

    @overload
    def make(
        self, task_id: str, env_type: Literal["dm"], **kwargs: Any
    ) -> DMEnvPool: ...

    @overload
    def make(
        self, task_id: str, env_type: Literal["gymnasium"], **kwargs: Any
    ) -> GymnasiumEnvPool: ...

    @overload
    def make(
        self, task_id: str, env_type: str, **kwargs: Any
    ) -> DMEnvPool | GymnasiumEnvPool: ...

    def make(
        self, task_id: str, env_type: str, **kwargs: Any
    ) -> DMEnvPool | GymnasiumEnvPool:
        """Make envpool."""
        from_pixels, wrapper_kwargs = self._extract_make_options(kwargs)
        if "gym_reset_return_info" not in kwargs:
            kwargs["gym_reset_return_info"] = True
        if not kwargs["gym_reset_return_info"]:
            raise ValueError(
                "EnvPool's gym API now follows gymnasium reset semantics and "
                "always returns an info dictionary "
                "after resets."
            )

        assert task_id in self.specs, (
            f"{task_id} is not supported, `envpool.list_all_envs()` may help."
        )
        assert env_type in ["dm", "gymnasium"]

        spec = self._make_env_spec(task_id, from_pixels=from_pixels, **kwargs)
        import_path, envpool_cls = self._resolve_envpool_entry(
            task_id, env_type, from_pixels
        )
        env = getattr(importlib.import_module(import_path), envpool_cls)(spec)
        return self._apply_wrapper_kwargs(env, wrapper_kwargs)

    def make_dm(self, task_id: str, **kwargs: Any) -> DMEnvPool:
        """Make dm_env compatible envpool."""
        return self.make(task_id, "dm", **kwargs)

    def make_gymnasium(self, task_id: str, **kwargs: Any) -> GymnasiumEnvPool:
        """Make gymnasium.Env compatible envpool."""
        return self.make(task_id, "gymnasium", **kwargs)

    def make_spec(self, task_id: str, **make_kwargs: Any) -> EnvSpec:
        """Make EnvSpec."""
        from_pixels, _ = self._extract_make_options(make_kwargs)
        return self._make_env_spec(
            task_id, from_pixels=from_pixels, **make_kwargs
        )

    @staticmethod
    def _assert_int32_seed(seed: Any) -> None:
        INT_MAX = 2**31
        assert -INT_MAX <= seed < INT_MAX, (
            f"Seed should be in range of int32, got {seed}"
        )

    @staticmethod
    def _is_env_seed_sequence(seed: Any) -> bool:
        return (
            isinstance(seed, Sequence) and not isinstance(seed, str | bytes)
        ) or isinstance(seed, np.ndarray)

    def _normalize_env_seed(self, seed: Any, num_envs: int) -> list[int]:
        if isinstance(seed, np.ndarray):
            assert seed.ndim == 1, (
                "`seed` as an array must be 1-dimensional, "
                f"got shape {seed.shape}"
            )
            seed = seed.tolist()
        else:
            seed = list(seed)
        assert len(seed) == num_envs, (
            "When `seed` is a sequence, its length must match `num_envs`, "
            f"got len(seed) = {len(seed)} and num_envs = {num_envs}"
        )
        normalized_seed = [int(s) for s in seed]
        for s in normalized_seed:
            self._assert_int32_seed(s)
        return normalized_seed

    def list_all_envs(self) -> list[str]:
        """Return all available task_id."""
        return list(self.specs.keys())


# use a global EnvRegistry
registry = EnvRegistry()
register = registry.register


@overload
def make(task_id: str, env_type: Literal["dm"], **kwargs: Any) -> DMEnvPool: ...


@overload
def make(
    task_id: str, env_type: Literal["gym"], **kwargs: Any
) -> GymnasiumEnvPool: ...


@overload
def make(
    task_id: str, env_type: Literal["gymnasium"], **kwargs: Any
) -> GymnasiumEnvPool: ...


@overload
def make(
    task_id: str, env_type: str, **kwargs: Any
) -> DMEnvPool | GymnasiumEnvPool: ...


def make(
    task_id: str, env_type: str, **kwargs: Any
) -> DMEnvPool | GymnasiumEnvPool:
    """Make an EnvPool with a public, typed interface."""
    if env_type == "dm":
        return registry.make(task_id, "dm", **kwargs)
    if env_type in ("gym", "gymnasium"):
        return registry.make(task_id, "gymnasium", **kwargs)
    raise AssertionError(
        "env_type should be one of 'dm', 'gym', or 'gymnasium'."
    )


def make_dm(task_id: str, **kwargs: Any) -> DMEnvPool:
    """Make dm_env compatible envpool."""
    return registry.make_dm(task_id, **kwargs)


def make_gym(task_id: str, **kwargs: Any) -> GymnasiumEnvPool:
    """Make gym.Env compatible envpool."""
    return make_gymnasium(task_id, **kwargs)


def make_gymnasium(task_id: str, **kwargs: Any) -> GymnasiumEnvPool:
    """Make gymnasium.Env compatible envpool."""
    return registry.make_gymnasium(task_id, **kwargs)


def make_spec(task_id: str, **kwargs: Any) -> EnvSpec:
    """Make an EnvSpec with a typed public interface."""
    return registry.make_spec(task_id, **kwargs)


def list_all_envs() -> list[str]:
    """Return all registered environment ids."""
    return registry.list_all_envs()
