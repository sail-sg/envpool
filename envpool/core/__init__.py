# Copyright 2023 Garena Online Private Limited
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
"""Core Python bindings exposed by envpool.

This module currently exports the shared worker-pool handle used by
`envpool.make_thread_pool(...)` and the optional `thread_pool=` constructor
argument on envpool instances.
"""

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

try:
    from .shared_thread_pool import _SharedThreadPool
except ModuleNotFoundError:
    _shared_thread_pool_dll = (
        Path(__file__).resolve().with_name("shared_thread_pool_dll.dll")
    )
    if not _shared_thread_pool_dll.is_file():
        raise
    _shared_thread_pool_name = f"{__name__}.shared_thread_pool"
    _shared_thread_pool_loader = importlib.machinery.ExtensionFileLoader(
        _shared_thread_pool_name,
        str(_shared_thread_pool_dll),
    )
    _shared_thread_pool_spec = importlib.util.spec_from_loader(
        _shared_thread_pool_name,
        _shared_thread_pool_loader,
        origin=str(_shared_thread_pool_dll),
    )
    if (
        _shared_thread_pool_spec is None
        or _shared_thread_pool_spec.loader is None
    ):
        raise
    _shared_thread_pool = importlib.util.module_from_spec(
        _shared_thread_pool_spec
    )
    sys.modules[_shared_thread_pool_name] = _shared_thread_pool
    _shared_thread_pool_spec.loader.exec_module(_shared_thread_pool)
    _SharedThreadPool = _shared_thread_pool._SharedThreadPool

SharedThreadPool = _SharedThreadPool

__all__ = [
    "SharedThreadPool",
]
