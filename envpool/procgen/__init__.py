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
"""Procgen env Init."""

import os
import sys

from envpool.python.api import py_env

_WINDOWS_DLL_DIR_HANDLES: list[object] = []
_PROCGEN_EXPORTS = {
    "ProcgenEnvSpec",
    "ProcgenDMEnvPool",
    "ProcgenGymnasiumEnvPool",
}
_LINUX_QT_RUNTIME_ERROR = (
    "EnvPool Procgen requires the system Qt 5 runtime on Linux. "
    "Install the Qt5 Core/Gui shared libraries, for example via "
    "`apt install qtbase5-dev` or `dnf install qt5-qtbase`."
)
_PROCGEN_IMPORT_ERROR: ImportError | None = None


def _is_linux_qt_import_error(exc: ImportError) -> bool:
    return sys.platform.startswith("linux") and (
        "libQt5Core" in str(exc) or "libQt5Gui" in str(exc)
    )


def _raise_procgen_import_error() -> None:
    assert _PROCGEN_IMPORT_ERROR is not None
    raise ImportError(_LINUX_QT_RUNTIME_ERROR) from _PROCGEN_IMPORT_ERROR


if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    # Procgen links against Qt on Windows, so register the Qt bin dir before
    # importing the extension module.
    for env_var in ("QT_ROOT_DIR", "BAZEL_RULES_QT_DIR"):
        qt_root = os.environ.get(env_var)
        if not qt_root:
            continue
        qt_bin = os.path.join(qt_root, "bin")
        if os.path.isdir(qt_bin):
            _WINDOWS_DLL_DIR_HANDLES.append(os.add_dll_directory(qt_bin))
            break

try:
    from .procgen_envpool import (
        _ProcgenEnvPool,
        _ProcgenEnvSpec,
    )
except ImportError as exc:
    if not _is_linux_qt_import_error(exc):
        raise
    _PROCGEN_IMPORT_ERROR = exc
else:
    (
        ProcgenEnvSpec,
        ProcgenDMEnvPool,
        ProcgenGymnasiumEnvPool,
    ) = py_env(_ProcgenEnvSpec, _ProcgenEnvPool)


def __getattr__(name: str) -> object:
    if name in _PROCGEN_EXPORTS and _PROCGEN_IMPORT_ERROR is not None:
        _raise_procgen_import_error()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ProcgenEnvSpec",
    "ProcgenDMEnvPool",
    "ProcgenGymnasiumEnvPool",
]
