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
"""Protocol of C++ EnvPool."""

from typing import (
    Any,
    Callable,
    Literal,
    Sequence,
    overload,
)

import numpy as np
from dm_env import TimeStep
from typing_extensions import Protocol, TypeAlias

ObsType: TypeAlias = Any
InfoDict: TypeAlias = dict[str, Any]
ActionInput: TypeAlias = dict[str, Any] | np.ndarray
RenderEnvId: TypeAlias = int | Sequence[int] | np.ndarray | None
GymResetReturn: TypeAlias = tuple[ObsType, InfoDict]
GymStepReturn: TypeAlias = tuple[ObsType, Any, Any, Any, InfoDict]
GymnasiumResetReturn: TypeAlias = tuple[ObsType, InfoDict]
GymnasiumStepReturn: TypeAlias = tuple[ObsType, Any, Any, Any, InfoDict]


class EnvSpec(Protocol):
    """Cpp EnvSpec class."""

    _config_keys: list[str]
    _default_config_values: tuple
    gen_config: type

    def __init__(self, config: tuple):
        """Protocol for constructor of EnvSpec."""

    @property
    def _state_spec(self) -> tuple:
        """Cpp private _state_spec."""

    @property
    def _action_spec(self) -> tuple:
        """Cpp private _action_spec."""

    @property
    def _state_keys(self) -> list:
        """Cpp private _state_keys."""

    @property
    def _action_keys(self) -> list:
        """Cpp private _action_keys."""

    @property
    def _config_values(self) -> tuple:
        """Cpp private _config_values."""

    @property
    def config(self) -> Any:
        """Configuration used to create the current EnvSpec."""

    @property
    def state_array_spec(self) -> dict[str, Any]:
        """Specs of the states of the environment in ArraySpec format."""

    @property
    def action_array_spec(self) -> dict[str, Any]:
        """Specs of the actions of the environment in ArraySpec format."""

    def observation_spec(self) -> Any:
        """Specs of the observations of the environment in dm_env format."""

    def action_spec(self) -> Any:
        """Specs of the actions of the environment in dm_env format."""

    @property
    def observation_space(self) -> Any:
        """Specs of the observations of the environment in gym.Env format."""

    @property
    def action_space(self) -> Any:
        """Specs of the actions of the environment in gym.Env format."""

    @property
    def gymnasium_observation_space(
        self,
    ) -> Any:
        """Specs of the observations of the environment in gymnasium format."""

    @property
    def gymnasium_action_space(
        self,
    ) -> Any:
        """Specs of the actions of the environment in gymnasium format."""

    @property
    def reward_threshold(self) -> float | None:
        """Reward threshold, None for no threshold."""


class ArraySpec:
    """Spec of numpy array."""

    def __init__(
        self,
        dtype: type,
        shape: list[int],
        bounds: tuple[Any, Any],
        element_wise_bounds: tuple[Any, Any],
        is_discrete: bool = False,
    ):
        """Constructor of ArraySpec."""
        self.dtype = dtype
        self.shape = shape
        self.is_discrete = is_discrete
        if element_wise_bounds[0]:
            self.minimum = np.array(element_wise_bounds[0])
        else:
            self.minimum = bounds[0]
        if element_wise_bounds[1]:
            self.maximum = np.array(element_wise_bounds[1])
        else:
            self.maximum = bounds[1]

    def __repr__(self) -> str:
        """Beautify debug info."""
        return (
            f"ArraySpec(shape={self.shape}, dtype={self.dtype}, "
            f"minimum={self.minimum}, maximum={self.maximum})"
        )


class EnvPool(Protocol):
    """Cpp PyEnvpool class interface."""

    _state_keys: list[str]
    _state_names: list[str]
    _action_keys: list[str]
    _action_names: list[str]
    _check_action_finished: bool
    _all_env_ids: np.ndarray
    _last_action_name: str
    _last_action_type: Any
    _last_players_env_id: np.ndarray
    spec: Any

    def __init__(self, spec: EnvSpec):
        """Constructor of EnvPool."""

    def __len__(self) -> int:
        """Return the number of environments."""

    @property
    def _spec(self) -> EnvSpec:
        """Cpp env spec."""

    @property
    def _action_spec(self) -> list:
        """Cpp action spec."""

    def _check_action(self, actions: list) -> None:
        """Check action shapes."""

    def _player_action_count(self, adict: dict[str, Any]) -> int | None:
        """Infer the leading player-action dimension."""

    def _cached_players_env_id(
        self, env_id: np.ndarray, player_count: int
    ) -> np.ndarray | None:
        """Reuse cached player-to-env mapping when available."""

    def _infer_players_env_id(self, adict: dict[str, Any]) -> np.ndarray:
        """Infer players.env_id for simplified multiplayer actions."""

    def _recv(self) -> list[np.ndarray]:
        """Cpp private _recv method."""

    def _send(self, action: list[np.ndarray]) -> None:
        """Cpp private _send method."""

    def _reset(self, env_id: np.ndarray) -> None:
        """Cpp private _reset method."""

    def _render(
        self,
        env_ids: np.ndarray,
        width: int,
        height: int,
        camera_id: int,
    ) -> np.ndarray:
        """Cpp private _render method."""

    def _render_config(self) -> tuple[str | None, int, int, int, int]:
        """Normalized render config."""

    def _pixel_observation_render_size(self) -> tuple[int, int] | None:
        """Resolved render size for native pixel observations when enabled."""

    def _ensure_pixel_observation_context(self) -> None:
        """Prepare a platform render context for native pixel observations."""

    def _ensure_platform_render_context(self, width: int, height: int) -> None:
        """Prepare any platform-specific GL context required before render."""

    def _show_human_frame(self, frame: np.ndarray) -> None:
        """Display a rendered frame in a Python viewer."""

    _render_window_name: str
    _render_window_open: bool

    def _from(
        self,
        action: dict[str, Any] | np.ndarray,
        env_id: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Convertion for input action."""

    def _to(
        self,
        state: list[np.ndarray],
        reset: bool,
        return_info: bool,
    ) -> Any:
        """A switch of to_dm and to_gym for output state."""

    @property
    def all_env_ids(self) -> np.ndarray:
        """All env_id in numpy ndarray with dtype=np.int32."""

    @property
    def is_async(self) -> bool:
        """Return if this env is in sync mode or async mode."""

    @property
    def observation_space(self) -> Any:
        """Gym observation space."""

    @property
    def action_space(self) -> Any:
        """Gym action space."""

    def observation_spec(self) -> Any:
        """Dm observation spec."""

    def action_spec(self) -> Any:
        """Dm action spec."""

    def seed(self, seed: int | list[int] | None = None) -> None:
        """Set the seed for all environments."""

    @property
    def config(self) -> dict[str, Any]:
        """Envpool config."""

    def send(
        self,
        action: ActionInput,
        env_id: np.ndarray | None = None,
    ) -> None:
        """Envpool send wrapper."""

    @overload
    def recv(
        self,
        reset: Literal[False] = False,
        return_info: bool = True,
    ) -> Any: ...

    @overload
    def recv(
        self,
        reset: Literal[True],
        return_info: bool = True,
    ) -> Any: ...

    def recv(
        self,
        reset: bool = False,
        return_info: bool = True,
    ) -> Any:
        """Envpool recv wrapper."""

    def async_reset(self) -> None:
        """Envpool async reset interface."""

    def step(
        self,
        action: ActionInput,
        env_id: np.ndarray | None = None,
    ) -> Any:
        """Envpool step interface that performs send/recv."""

    def reset(
        self,
        env_id: np.ndarray | None = None,
    ) -> Any:
        """Envpool reset interface."""

    def render(
        self,
        env_ids: RenderEnvId = None,
        camera_id: int | None = None,
    ) -> np.ndarray | None:
        """Render one or more envs."""

    def close(self) -> None:
        """Close the underlying environment."""

    def xla(self) -> tuple[Any, Callable, Callable, Callable]:
        """Get the xla functions."""


class DMEnvPool(EnvPool, Protocol):
    """dm_env-compatible EnvPool interface."""

    def observation_spec(self) -> Any:
        """Dm observation spec."""

    def action_spec(self) -> Any:
        """Dm action spec."""

    def recv(
        self,
        reset: bool = False,
        return_info: bool = True,
    ) -> TimeStep:
        """Recv a dm_env timestep."""

    def step(
        self,
        action: ActionInput,
        env_id: np.ndarray | None = None,
    ) -> TimeStep:
        """Step the dm_env-compatible envpool."""

    def reset(
        self,
        env_id: np.ndarray | None = None,
    ) -> TimeStep:
        """Reset the dm_env-compatible envpool."""


class GymEnvPool(EnvPool, Protocol):
    """gym-compatible EnvPool interface."""

    @property
    def observation_space(self) -> Any:
        """Gym observation space."""

    @property
    def action_space(self) -> Any:
        """Gym action space."""

    @overload
    def recv(
        self,
        reset: Literal[False] = False,
        return_info: bool = True,
    ) -> GymStepReturn: ...

    @overload
    def recv(
        self,
        reset: Literal[True],
        return_info: bool = True,
    ) -> GymResetReturn: ...

    @overload
    def recv(
        self,
        reset: bool = False,
        return_info: bool = True,
    ) -> GymResetReturn | GymStepReturn: ...

    def recv(
        self,
        reset: bool = False,
        return_info: bool = True,
    ) -> GymResetReturn | GymStepReturn:
        """Recv a gym observation/reset tuple."""

    def step(
        self,
        action: ActionInput,
        env_id: np.ndarray | None = None,
    ) -> GymStepReturn:
        """Step the gym-compatible envpool."""

    def reset(
        self,
        env_id: np.ndarray | None = None,
    ) -> GymResetReturn:
        """Reset the gym-compatible envpool."""


class GymnasiumEnvPool(EnvPool, Protocol):
    """gymnasium-compatible EnvPool interface."""

    @property
    def observation_space(self) -> Any:
        """Gymnasium observation space."""

    @property
    def action_space(self) -> Any:
        """Gymnasium action space."""

    @overload
    def recv(
        self,
        reset: Literal[False] = False,
        return_info: bool = True,
    ) -> GymnasiumStepReturn: ...

    @overload
    def recv(
        self,
        reset: Literal[True],
        return_info: bool = True,
    ) -> GymnasiumResetReturn: ...

    @overload
    def recv(
        self,
        reset: bool = False,
        return_info: bool = True,
    ) -> GymnasiumResetReturn | GymnasiumStepReturn: ...

    def recv(
        self,
        reset: bool = False,
        return_info: bool = True,
    ) -> GymnasiumResetReturn | GymnasiumStepReturn:
        """Recv a gymnasium observation/reset tuple."""

    def step(
        self,
        action: ActionInput,
        env_id: np.ndarray | None = None,
    ) -> GymnasiumStepReturn:
        """Step the gymnasium-compatible envpool."""

    def reset(
        self,
        env_id: np.ndarray | None = None,
    ) -> GymnasiumResetReturn:
        """Reset the gymnasium-compatible envpool."""
