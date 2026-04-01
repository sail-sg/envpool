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

import threading

_CONTEXT_LOCK = threading.Lock()
_GLFW_CONTEXT: "_GlfwContext | None" = None


class _GlfwContext:
    """Hidden GLFW window aligned with MuJoCo's upstream Windows backend."""

    def __init__(self, width: int, height: int) -> None:
        try:
            import glfw
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo rendering on Windows requires the `glfw` package."
            ) from exc
        if not glfw.init():
            raise RuntimeError("failed to initialize GLFW for MuJoCo render")
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
            raise RuntimeError("failed to create GLFW window for MuJoCo render")

    def make_current(self) -> None:
        self._glfw.make_context_current(self._window)


def ensure_mujoco_glfw_context(width: int, height: int) -> None:
    """Create a process-global hidden GLFW context and make it current."""
    global _GLFW_CONTEXT
    with _CONTEXT_LOCK:
        if _GLFW_CONTEXT is None:
            _GLFW_CONTEXT = _GlfwContext(width, height)
        _GLFW_CONTEXT.make_current()
