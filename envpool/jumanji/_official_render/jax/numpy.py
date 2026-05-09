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
"""NumPy-backed stand-in for the small jax.numpy viewer surface."""

from __future__ import annotations

from typing import Any

from numpy import *  # noqa: F403,F401

import numpy as _np

_UNSET = object()


def _jax_dtype(value: Any, dtype: Any = None) -> Any:
    if dtype is not None:
        return dtype
    array = _np.asarray(value)
    if array.dtype == _np.dtype("float64"):
        return _np.float32
    if array.dtype == _np.dtype("int64"):
        return _np.int32
    return None


class _AtIndexer:
    def __init__(self, array: "JaxArray") -> None:
        self._array = array
        self._index: Any = slice(None)

    def __getitem__(self, index: Any) -> "_AtIndexer":
        self._index = index
        return self

    def set(self, value: Any) -> "JaxArray":
        out = _np.array(self._array, copy=True)
        out[self._index] = value
        return _wrap(out)

    def add(self, value: Any) -> "JaxArray":
        out = _np.array(self._array, copy=True)
        out[self._index] += value
        return _wrap(out)


class JaxArray(_np.ndarray):
    __array_priority__ = 1000

    @property
    def at(self) -> _AtIndexer:
        return _AtIndexer(self)

    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        inputs = tuple(_unwrap(value) for value in inputs)
        kwargs = {key: _unwrap(value) for key, value in kwargs.items()}
        result = getattr(ufunc, method)(*inputs, **kwargs)
        if isinstance(result, tuple):
            return tuple(_wrap(value) for value in result)
        return _wrap(result)


def _unwrap(value: Any) -> Any:
    if isinstance(value, JaxArray):
        return value.view(_np.ndarray)
    if isinstance(value, tuple):
        return tuple(_unwrap(item) for item in value)
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    return value


def _wrap(value: Any) -> Any:
    if isinstance(value, _np.ndarray):
        if value.dtype == _np.dtype("float64"):
            value = value.astype(_np.float32)
        elif value.dtype == _np.dtype("int64"):
            value = value.astype(_np.int32)
    if isinstance(value, _np.ndarray) and not isinstance(value, JaxArray):
        return value.view(JaxArray)
    return value


def asarray(a: Any, dtype: Any = None, order: Any = None) -> JaxArray:
    dtype = _jax_dtype(a, dtype)
    return _wrap(_np.asarray(a, dtype=dtype, order=order))


def array(
    object: Any, dtype: Any = None, copy: bool = True, order: Any = "K", ndmin: int = 0
) -> JaxArray:
    dtype = _jax_dtype(object, dtype)
    return _wrap(_np.array(object, dtype=dtype, copy=copy, order=order, ndmin=ndmin))


def zeros(shape: Any, dtype: Any = float, order: str = "C") -> JaxArray:
    if dtype is float:
        dtype = _np.float32
    return _wrap(_np.zeros(shape, dtype=dtype, order=order))


def zeros_like(a: Any, dtype: Any = None, order: str = "K", subok: bool = True) -> JaxArray:
    return _wrap(_np.zeros_like(a, dtype=dtype, order=order, subok=subok))


def ones(shape: Any, dtype: Any = None, order: str = "C") -> JaxArray:
    if dtype is None:
        dtype = _np.float32
    return _wrap(_np.ones(shape, dtype=dtype, order=order))


def ones_like(a: Any, dtype: Any = None, order: str = "K", subok: bool = True) -> JaxArray:
    return _wrap(_np.ones_like(a, dtype=dtype, order=order, subok=subok))


def full(shape: Any, fill_value: Any, dtype: Any = None, order: str = "C") -> JaxArray:
    dtype = _jax_dtype(fill_value, dtype)
    return _wrap(_np.full(shape, fill_value, dtype=dtype, order=order))


def full_like(
    a: Any,
    fill_value: Any,
    dtype: Any = None,
    order: str = "K",
    subok: bool = True,
    shape: Any = _UNSET,
) -> JaxArray:
    dtype = _jax_dtype(a, dtype)
    kwargs = {} if shape is _UNSET else {"shape": shape}
    return _wrap(
        _np.full_like(
            a, fill_value, dtype=dtype, order=order, subok=subok, **kwargs
        )
    )


def arange(*args: Any, **kwargs: Any) -> JaxArray:
    if "dtype" not in kwargs:
        kwargs["dtype"] = _np.float32 if any(isinstance(arg, float) for arg in args) else _np.int32
    return _wrap(_np.arange(*args, **kwargs))


def concatenate(seq: Any, axis: int = 0, out: Any = None, dtype: Any = None, casting: str = "same_kind") -> JaxArray:
    return _wrap(_np.concatenate(seq, axis=axis, out=out, dtype=dtype, casting=casting))


def stack(arrays: Any, axis: int = 0, out: Any = None, dtype: Any = None, casting: str = "same_kind") -> JaxArray:
    return _wrap(_np.stack(arrays, axis=axis, out=out, dtype=dtype, casting=casting))


def vstack(tup: Any, *, dtype: Any = None, casting: str = "same_kind") -> JaxArray:
    return _wrap(_np.vstack(tup, dtype=dtype, casting=casting))


def where(condition: Any, x: Any = _UNSET, y: Any = _UNSET) -> Any:
    if x is _UNSET and y is _UNSET:
        return _np.where(condition)
    return _wrap(_np.where(condition, x, y))


def power(x1: Any, x2: Any, *args: Any, **kwargs: Any) -> JaxArray:
    return _wrap(_np.power(x1, x2, *args, **kwargs))


def flipud(m: Any) -> JaxArray:
    return _wrap(_np.flipud(m))


def maximum(x1: Any, x2: Any, *args: Any, **kwargs: Any) -> JaxArray:
    return _wrap(_np.maximum(x1, x2, *args, **kwargs))


def minimum(x1: Any, x2: Any, *args: Any, **kwargs: Any) -> JaxArray:
    return _wrap(_np.minimum(x1, x2, *args, **kwargs))


Array = JaxArray
ndarray = JaxArray
newaxis = _np.newaxis
