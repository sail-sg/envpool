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

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


def _load_shared_thread_pool_from_runfiles_manifest() -> Any:
    manifest_path = os.environ.get("RUNFILES_MANIFEST_FILE")
    runfiles_path = Path(__file__).resolve()
    runfile_key = "envpool/envpool/core/shared_thread_pool.pyd"
    for parent in runfiles_path.parents:
        if parent.name.endswith(".runfiles"):
            manifest_path = manifest_path or str(parent / "MANIFEST")
            runfile_key = (
                runfiles_path.with_name("shared_thread_pool.pyd")
                .relative_to(parent)
                .as_posix()
            )
            break
    if manifest_path is None:
        raise ModuleNotFoundError("Cannot locate Bazel runfiles MANIFEST.")

    shared_thread_pool_path: Path | None = None
    with Path(manifest_path).open(encoding="utf-8") as manifest_file:
        for line in manifest_file:
            manifest_key, _, manifest_value = line.rstrip("\n").partition(" ")
            if manifest_key == runfile_key:
                shared_thread_pool_path = Path(manifest_value)
                break
    if shared_thread_pool_path is None:
        raise ModuleNotFoundError(
            f"Cannot find {runfile_key!r} in {manifest_path!r}."
        )

    module_name = f"{__name__}.shared_thread_pool"
    loader = importlib.machinery.ExtensionFileLoader(
        module_name,
        str(shared_thread_pool_path),
    )
    spec = importlib.util.spec_from_loader(
        module_name,
        loader,
        origin=str(shared_thread_pool_path),
    )
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(
            f"Cannot load shared_thread_pool from {shared_thread_pool_path}."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module._SharedThreadPool


try:
    from .shared_thread_pool import _SharedThreadPool
except ModuleNotFoundError:
    _SharedThreadPool = _load_shared_thread_pool_from_runfiles_manifest()

SharedThreadPool = _SharedThreadPool

__all__ = [
    "SharedThreadPool",
]
