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
from typing import Any, Dict, List, Tuple

import gym
from packaging import version

base_path = os.path.abspath(os.path.dirname(__file__))


class EnvRegistry:
  """A collection of available envs."""

  def __init__(self) -> None:
    """Constructor of EnvRegistry."""
    self.specs: Dict[str, Tuple[str, str, Dict[str, Any]]] = {}
    self.envpools: Dict[str, Dict[str, Tuple[str, str]]] = {}

  def register(
    self, task_id: str, import_path: str, spec_cls: str, dm_cls: str,
    gym_cls: str, gymnasium_cls: str, **kwargs: Any
  ) -> None:
    """Register EnvSpec and EnvPool in global EnvRegistry."""
    assert task_id not in self.specs
    if "base_path" not in kwargs:
      kwargs["base_path"] = base_path
    self.specs[task_id] = (import_path, spec_cls, kwargs)
    self.envpools[task_id] = {
      "dm": (import_path, dm_cls),
      "gym": (import_path, gym_cls),
      "gymnasium": (import_path, gymnasium_cls)
    }

  def make(self, task_id: str, env_type: str, **kwargs: Any) -> Any:
    """Make envpool."""
    new_gym_api = version.parse(gym.__version__) >= version.parse("0.26.0")
    if "gym_reset_return_info" not in kwargs:
      kwargs["gym_reset_return_info"] = new_gym_api
    if new_gym_api and not kwargs["gym_reset_return_info"]:
      raise ValueError(
        "You are using gym>=0.26.0 but passed `gym_reset_return_info=False`. "
        "The new gym API requires environments to return an info dictionary "
        "after resets."
      )

    assert task_id in self.specs, \
      f"{task_id} is not supported, `envpool.list_all_envs()` may help."
    assert env_type in ["dm", "gym", "gymnasium"]

    spec = self.make_spec(task_id, **kwargs)
    import_path, envpool_cls = self.envpools[task_id][env_type]
    return getattr(importlib.import_module(import_path), envpool_cls)(spec)

  def make_dm(self, task_id: str, **kwargs: Any) -> Any:
    """Make dm_env compatible envpool."""
    return self.make(task_id, "dm", **kwargs)

  def make_gym(self, task_id: str, **kwargs: Any) -> Any:
    """Make gym.Env compatible envpool."""
    return self.make(task_id, "gym", **kwargs)

  def make_gymnasium(self, task_id: str, **kwargs: Any) -> Any:
    """Make gymnasium.Env compatible envpool."""
    return self.make(task_id, "gymnasium", **kwargs)

  def make_spec(self, task_id: str, **make_kwargs: Any) -> Any:
    """Make EnvSpec."""
    import_path, spec_cls, kwargs = self.specs[task_id]
    kwargs = {**kwargs, **make_kwargs}

    # check arguments
    if "seed" in kwargs:  # Issue 214
      INT_MAX = 2**31
      assert -INT_MAX <= kwargs["seed"] < INT_MAX, \
        f"Seed should be in range of int32, got {kwargs['seed']}"
    if "num_envs" in kwargs:
      assert kwargs["num_envs"] >= 1
    if "batch_size" in kwargs:
      assert 0 <= kwargs["batch_size"] <= kwargs["num_envs"]
    if "max_num_players" in kwargs:
      assert 1 <= kwargs["max_num_players"]

    spec_cls = getattr(importlib.import_module(import_path), spec_cls)
    config = spec_cls.gen_config(**kwargs)
    return spec_cls(config)

  def list_all_envs(self) -> List[str]:
    """Return all available task_id."""
    return list(self.specs.keys())


# use a global EnvRegistry
registry = EnvRegistry()
register = registry.register
make = registry.make
make_dm = registry.make_dm
make_gym = registry.make_gym
make_gymnasium = registry.make_gymnasium
make_spec = registry.make_spec
list_all_envs = registry.list_all_envs
