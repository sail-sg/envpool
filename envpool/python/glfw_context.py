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
"""GLFW-backed MuJoCo render context helpers for Windows."""

from __future__ import annotations

import ctypes
import os
import threading
import warnings
from pathlib import Path

_CONTEXT_LOCK = threading.Lock()
_GLFW_CONTEXT: "_GlfwContext | None" = None
_GLFW_FAILURE_REASON: str | None = None
_WINDOWS_DLL_HANDLES: list[object] = []
_REGISTERED_DLL_DIRS: set[str] = set()
_PRELOADED_DLL_PATHS: set[str] = set()


def preload_windows_gl_dlls(
    *, prepend_path: bool = True, strict: bool = False
) -> None:
    """Preload a Windows OpenGL userspace stack from ``ENVPOOL_DLL_DIR``."""
    if not hasattr(os, "add_dll_directory"):
        return
    dll_dir = os.environ.get("ENVPOOL_DLL_DIR")
    if not dll_dir:
        return
    resolved_dir = Path(dll_dir).expanduser().resolve()
    if not resolved_dir.is_dir():
        if strict:
            raise FileNotFoundError(
                f"ENVPOOL_DLL_DIR does not exist: {resolved_dir}"
            )
        return
    resolved_str = str(resolved_dir)
    if prepend_path:
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if resolved_str not in path_entries:
            filtered_entries = [entry for entry in path_entries if entry]
            os.environ["PATH"] = os.pathsep.join([
                resolved_str,
                *filtered_entries,
            ])
    if resolved_str not in _REGISTERED_DLL_DIRS:
        _WINDOWS_DLL_HANDLES.append(os.add_dll_directory(resolved_str))
        _REGISTERED_DLL_DIRS.add(resolved_str)
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        return
    for dll_name in ("libglapi.dll", "libgallium_wgl.dll", "opengl32.dll"):
        dll_path = resolved_dir / dll_name
        dll_path_str = str(dll_path)
        if dll_path.is_file() and dll_path_str not in _PRELOADED_DLL_PATHS:
            _WINDOWS_DLL_HANDLES.append(win_dll(str(dll_path)))
            _PRELOADED_DLL_PATHS.add(dll_path_str)


def _glfw_error_details(glfw: object) -> str:
    get_error = getattr(glfw, "get_error", None)
    if get_error is None:
        return ""
    error = get_error()
    if not isinstance(error, tuple) or len(error) != 2:
        return ""
    error_code, description = error
    if error_code is None:
        return ""
    return f": ({error_code}) {description!r}"


def _warn_glfw_fallback(reason: str) -> None:
    warnings.warn(
        "Failed to initialize the GLFW-backed MuJoCo render context on "
        f"Windows; falling back to EnvPool's native WGL path. {reason}",
        RuntimeWarning,
        stacklevel=2,
    )


class _GlfwContext:
    """Hidden GLFW window aligned with MuJoCo's upstream Windows backend."""

    def __init__(self, width: int, height: int) -> None:
        preload_windows_gl_dlls()
        try:
            import glfw
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo rendering on Windows requires the `glfw` package."
            ) from exc
        if not glfw.init():
            raise RuntimeError(
                "failed to initialize GLFW for MuJoCo render"
                f"{_glfw_error_details(glfw)}"
            )
        glfw.window_hint(glfw.VISIBLE, 0)
        self._glfw = glfw
        self._window = glfw.create_window(
            width=max(width, 1),
            height=max(height, 1),
            title="EnvPool MuJoCo Hidden Window",
            monitor=None,
            share=None,
        )
        if self._window is None:
            glfw.terminate()
            raise RuntimeError(
                "failed to create GLFW window for MuJoCo render"
                f"{_glfw_error_details(glfw)}"
            )

    def make_current(self) -> None:
        if self._window is None:
            raise RuntimeError("GLFW MuJoCo render window is unavailable")
        self._glfw.make_context_current(self._window)

    def close(self) -> None:
        if self._window is None:
            return
        if self._glfw.get_current_context() == self._window:
            self._glfw.make_context_current(None)
        self._glfw.destroy_window(self._window)
        self._window = None
        self._glfw.terminate()


def try_ensure_mujoco_glfw_context(width: int, height: int) -> bool:
    """Best-effort GLFW context setup that falls back to native WGL."""
    global _GLFW_CONTEXT, _GLFW_FAILURE_REASON
    with _CONTEXT_LOCK:
        if _GLFW_FAILURE_REASON is not None:
            return False
        if _GLFW_CONTEXT is None:
            try:
                _GLFW_CONTEXT = _GlfwContext(width, height)
            except Exception as exc:
                _GLFW_FAILURE_REASON = str(exc)
                _warn_glfw_fallback(_GLFW_FAILURE_REASON)
                return False
        try:
            _GLFW_CONTEXT.make_current()
        except Exception as exc:
            _GLFW_FAILURE_REASON = str(exc)
            if _GLFW_CONTEXT is not None:
                _GLFW_CONTEXT.close()
                _GLFW_CONTEXT = None
            _warn_glfw_fallback(_GLFW_FAILURE_REASON)
            return False
        return True


def ensure_mujoco_glfw_context(width: int, height: int) -> None:
    """Create a process-global hidden GLFW context and make it current."""
    if not try_ensure_mujoco_glfw_context(width, height):
        raise RuntimeError(
            "failed to initialize GLFW-backed MuJoCo render context"
        )
