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

"""Minimal `ml_collections.config_dict` for MuJoCo Playground oracle tests."""

from __future__ import annotations

from typing import Any


class ConfigDict(dict[str, Any]):
    """Dict with attribute access and flattened update support."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize from mapping or keyword values."""
        super().__init__()
        self.update(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Return config values through attribute access."""
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(name) from err

    def __setattr__(self, name: str, value: Any) -> None:
        """Store config values through attribute assignment."""
        self[name] = _convert(value)

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        """Update values while recursively converting nested dicts."""
        values = dict(*args, **kwargs)
        for key, value in values.items():
            self[key] = _convert(value)

    def copy(self) -> ConfigDict:
        """Return a shallow copy preserving ConfigDict conversion."""
        return ConfigDict(self)

    def lock(self) -> ConfigDict:
        """Match ml_collections lock API; this lightweight stub is mutable."""
        return self

    def update_from_flattened_dict(self, values: dict[str, Any]) -> None:
        """Update nested values from dotted keys."""
        for key, value in values.items():
            target: ConfigDict = self
            parts = key.split(".")
            for part in parts[:-1]:
                child = target.get(part)
                if not isinstance(child, ConfigDict):
                    child = ConfigDict()
                    target[part] = child
                target = child
            target[parts[-1]] = _convert(value)


def _convert(value: Any) -> Any:
    if isinstance(value, ConfigDict):
        return value
    if isinstance(value, dict):
        return ConfigDict(value)
    return value


def create(**kwargs: Any) -> ConfigDict:
    """Create a ConfigDict from keyword values."""
    return ConfigDict(kwargs)
