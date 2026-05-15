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

"""Minimal `flax.struct` surface needed by MuJoCo Playground oracle tests."""

import dataclasses
from collections.abc import Callable
from typing import Any, TypeVar, overload

_T = TypeVar("_T")


def _replace(self: Any, **kwargs: Any) -> Any:
    return dataclasses.replace(self, **kwargs)


@overload
def dataclass(cls: type[_T]) -> type[_T]: ...


@overload
def dataclass(cls: None = None) -> Callable[[type[_T]], type[_T]]: ...


def dataclass(cls: type[_T] | None = None) -> Any:
    """Decorate a class like `flax.struct.dataclass` for eager oracle code."""

    def wrap(klass: type[_T]) -> type[_T]:
        wrapped = dataclasses.dataclass(klass)
        wrapped.replace = _replace  # type: ignore[attr-defined]
        return wrapped

    if cls is None:
        return wrap
    return wrap(cls)
