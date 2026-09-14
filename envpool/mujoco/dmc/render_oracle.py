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
"""Shared GL bootstrap for the pinned dm_control render oracle."""

import ctypes
import platform
from typing import Any

from dm_control import _render
from dm_control._render import base as dm_control_render_base
from dm_control._render import executor as dm_control_render_executor


def configure_macos_dm_control_renderer() -> None:
    """Use the existing DMC CGL context setup on macOS."""
    if platform.system() != "Darwin":
        return

    class _CglContext(dm_control_render_base.ContextBase):
        def __init__(self, max_width: int, max_height: int):
            super().__init__(
                max_width,
                max_height,
                dm_control_render_executor.PassthroughRenderExecutor,
            )

        def _platform_init(self, max_width: int, max_height: int) -> None:
            del max_width, max_height
            from mujoco.cgl import cgl

            attrib = cgl.CGLPixelFormatAttribute
            profile = cgl.CGLOpenGLProfile
            attrib_values = (
                attrib.CGLPFAOpenGLProfile,
                profile.CGLOGLPVersion_Legacy,
                attrib.CGLPFAColorSize,
                24,
                attrib.CGLPFAAlphaSize,
                8,
                attrib.CGLPFADepthSize,
                24,
                attrib.CGLPFAStencilSize,
                8,
                attrib.CGLPFAAllowOfflineRenderers,
                0,
                0,  # terminator
            )
            attribs = (ctypes.c_int * len(attrib_values))(*attrib_values)
            self._pixel_format = cgl.CGLPixelFormatObj()
            num_pixel_formats = cgl.GLint()
            cgl.CGLChoosePixelFormat(
                attribs,
                ctypes.byref(self._pixel_format),
                ctypes.byref(num_pixel_formats),
            )
            if not self._pixel_format or num_pixel_formats.value == 0:
                raise RuntimeError("failed to create CGL pixel format")

            self._context = cgl.CGLContextObj()
            cgl.CGLCreateContext(
                self._pixel_format,
                0,
                ctypes.byref(self._context),
            )
            if not self._context:
                cgl.CGLReleasePixelFormat(self._pixel_format)
                self._pixel_format = None
                raise RuntimeError("failed to create CGL context")
            self._locked = False
            self._frame_settled = False

        def _platform_make_current(self) -> None:
            from mujoco.cgl import cgl

            cgl.CGLSetCurrentContext(self._context)
            # Mirror mujoco.cgl.GLContext so the official renderer uses the
            # same CGL lifecycle as EnvPool's native renderer.
            if not self._locked:
                cgl.CGLLockContext(self._context)
                self._locked = True

        def _platform_free(self) -> None:
            from mujoco.cgl import cgl

            if self._context:
                if self._locked:
                    cgl.CGLUnlockContext(self._context)
                    self._locked = False
                cgl.CGLSetCurrentContext(None)
                cgl.CGLReleaseContext(self._context)
                self._context = None
            if self._pixel_format:
                cgl.CGLReleasePixelFormat(self._pixel_format)
                self._pixel_format = None

    _render.Renderer = _CglContext
    _render.BACKEND = "cgl"
    _render.USING_GPU = True

    from dm_control.mujoco import engine

    render_on_gl_thread = engine.Camera._render_on_gl_thread

    def settled_render(camera: Any, depth: bool, overlays: Any) -> None:
        render_on_gl_thread(camera, depth, overlays)
        context = camera._physics.contexts.gl
        if not context._frame_settled:
            # Mirror OffscreenRenderer's existing four CGL settle passes. The
            # first MSAA readback can differ from every subsequent render of
            # the identical scene (including the CMU side camera). No physics,
            # task state, camera settings, or later frames are modified.
            for _ in range(4):
                render_on_gl_thread(camera, depth, overlays)
            context._frame_settled = True

    engine.Camera._render_on_gl_thread = settled_render
