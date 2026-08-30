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

"""Share the already-validated native GL bootstrap between MuJoCo oracles."""

import ctypes
import importlib
import platform
from typing import Any, ClassVar


def configure_macos_mujoco_renderer() -> None:
    """Use MuJoCo's default CGL pixel format with EnvPool's lock lifecycle."""
    if platform.system() != "Darwin":
        return

    import mujoco
    from mujoco import cgl as mujoco_cgl
    from mujoco import gl_context
    from mujoco.cgl import cgl
    from mujoco.rendering.classic import renderer as classic_renderer

    class _CglContext:
        def __init__(self, width: int, height: int) -> None:
            del width, height
            self._pixel_format: Any = None
            self._context: Any = None
            self._locked = False
            attrib = cgl.CGLPixelFormatAttribute
            profile = cgl.CGLOpenGLProfile
            preferred_attribs = (
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
                attrib.CGLPFAMultisample,
                attrib.CGLPFASampleBuffers,
                1,
                attrib.CGLPFASample,
                4,
                attrib.CGLPFAAccelerated,
                0,  # terminator
            )
            offline_attribs = (
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
                0,  # terminator
            )

            if not self._choose_pixel_format(
                cgl, preferred_attribs
            ) and not self._choose_pixel_format(cgl, offline_attribs):
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

        def _choose_pixel_format(
            self, cgl: Any, attrib_values: tuple[int, ...]
        ) -> bool:
            attribs = (ctypes.c_int * len(attrib_values))(*attrib_values)
            pixel_format = cgl.CGLPixelFormatObj()
            num_pixel_formats = cgl.GLint()
            try:
                cgl.CGLChoosePixelFormat(
                    attribs,
                    ctypes.byref(pixel_format),
                    ctypes.byref(num_pixel_formats),
                )
            except cgl.CGLError:
                return False
            if not pixel_format or num_pixel_formats.value == 0:
                return False
            self._pixel_format = pixel_format
            return True

        def make_current(self) -> None:
            cgl.CGLSetCurrentContext(self._context)
            if not self._locked:
                cgl.CGLLockContext(self._context)
                self._locked = True

        def free(self) -> None:
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

        def __del__(self) -> None:
            self.free()

    gl_context.GLContext = _CglContext
    mujoco.gl_context.GLContext = _CglContext
    mujoco_cgl.GLContext = _CglContext
    classic_renderer.gl_context.GLContext = _CglContext


def configure_windows_mujoco_renderer() -> None:
    """Match EnvPool's native WGL context for official render-oracle tests."""
    if platform.system() != "Windows" or getattr(
        configure_windows_mujoco_renderer, "_configured", False
    ):
        return

    import mujoco
    from mujoco import gl_context
    from mujoco import glfw as mujoco_glfw
    from mujoco.rendering.classic import gl_context as classic_gl_context
    from mujoco.rendering.classic import renderer as classic_renderer

    ctypes_attrs = vars(ctypes)
    wintypes = importlib.import_module("ctypes.wintypes")
    windll = ctypes_attrs["WinDLL"]
    winfunctype = ctypes_attrs["WINFUNCTYPE"]
    win_error = ctypes_attrs["WinError"]
    get_last_error = ctypes_attrs["get_last_error"]
    kernel32 = windll("kernel32", use_last_error=True)
    user32 = windll("user32", use_last_error=True)
    gdi32 = windll("gdi32", use_last_error=True)
    opengl32 = windll("opengl32", use_last_error=True)

    lresult = getattr(wintypes, "LRESULT", ctypes.c_ssize_t)
    hcursor = vars(wintypes).get("HCURSOR", wintypes.HANDLE)
    wndproc = winfunctype(
        lresult,
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    )
    user32.DefWindowProcW.argtypes = [
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    ]
    user32.DefWindowProcW.restype = lresult
    window_proc = wndproc(user32.DefWindowProcW)

    class _WndClass(ctypes.Structure):
        _fields_: ClassVar[Any] = [
            ("style", wintypes.UINT),
            ("lpfnWndProc", wndproc),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", wintypes.HINSTANCE),
            ("hIcon", wintypes.HICON),
            ("hCursor", hcursor),
            ("hbrBackground", wintypes.HBRUSH),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
        ]

    class _PixelFormatDescriptor(ctypes.Structure):
        _fields_: ClassVar[Any] = [
            ("nSize", wintypes.WORD),
            ("nVersion", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("iPixelType", ctypes.c_ubyte),
            ("cColorBits", ctypes.c_ubyte),
            ("cRedBits", ctypes.c_ubyte),
            ("cRedShift", ctypes.c_ubyte),
            ("cGreenBits", ctypes.c_ubyte),
            ("cGreenShift", ctypes.c_ubyte),
            ("cBlueBits", ctypes.c_ubyte),
            ("cBlueShift", ctypes.c_ubyte),
            ("cAlphaBits", ctypes.c_ubyte),
            ("cAlphaShift", ctypes.c_ubyte),
            ("cAccumBits", ctypes.c_ubyte),
            ("cAccumRedBits", ctypes.c_ubyte),
            ("cAccumGreenBits", ctypes.c_ubyte),
            ("cAccumBlueBits", ctypes.c_ubyte),
            ("cAccumAlphaBits", ctypes.c_ubyte),
            ("cDepthBits", ctypes.c_ubyte),
            ("cStencilBits", ctypes.c_ubyte),
            ("cAuxBuffers", ctypes.c_ubyte),
            ("iLayerType", ctypes.c_ubyte),
            ("bReserved", ctypes.c_ubyte),
            ("dwLayerMask", wintypes.DWORD),
            ("dwVisibleMask", wintypes.DWORD),
            ("dwDamageMask", wintypes.DWORD),
        ]

    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    user32.RegisterClassW.argtypes = [ctypes.POINTER(_WndClass)]
    user32.RegisterClassW.restype = wintypes.ATOM
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HWND,
        wintypes.HMENU,
        wintypes.HINSTANCE,
        wintypes.LPVOID,
    ]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.GetDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.DestroyWindow.restype = wintypes.BOOL
    gdi32.ChoosePixelFormat.argtypes = [
        wintypes.HDC,
        ctypes.POINTER(_PixelFormatDescriptor),
    ]
    gdi32.ChoosePixelFormat.restype = ctypes.c_int
    gdi32.SetPixelFormat.argtypes = [
        wintypes.HDC,
        ctypes.c_int,
        ctypes.POINTER(_PixelFormatDescriptor),
    ]
    gdi32.SetPixelFormat.restype = wintypes.BOOL
    opengl32.wglCreateContext.argtypes = [wintypes.HDC]
    opengl32.wglCreateContext.restype = ctypes.c_void_p
    opengl32.wglMakeCurrent.argtypes = [wintypes.HDC, ctypes.c_void_p]
    opengl32.wglMakeCurrent.restype = wintypes.BOOL
    opengl32.wglDeleteContext.argtypes = [ctypes.c_void_p]
    opengl32.wglDeleteContext.restype = wintypes.BOOL

    class _WglContext:
        _class_name = "EnvPoolMyoSuiteOracleOffscreen"
        _window_proc = window_proc
        _registered = False

        def __init__(self, width: int, height: int) -> None:
            del width, height
            self._window = None
            self._device_context = None
            self._context = None
            self._ensure_window_class()
            self._window = user32.CreateWindowExW(
                0,
                self._class_name,
                "EnvPool MyoSuite Oracle Offscreen",
                0x00CF0000,  # WS_OVERLAPPEDWINDOW
                0,
                0,
                1,
                1,
                None,
                None,
                kernel32.GetModuleHandleW(None),
                None,
            )
            if not self._window:
                raise win_error(get_last_error())
            self._device_context = user32.GetDC(self._window)
            if not self._device_context:
                self.free()
                raise win_error(get_last_error())
            pixel_format = _PixelFormatDescriptor()
            pixel_format.nSize = ctypes.sizeof(_PixelFormatDescriptor)
            pixel_format.nVersion = 1
            pixel_format.dwFlags = 0x00000004 | 0x00000020
            pixel_format.iPixelType = 0
            pixel_format.cColorBits = 24
            pixel_format.cAlphaBits = 8
            pixel_format.cDepthBits = 24
            pixel_format.cStencilBits = 8
            pixel_format.iLayerType = 0
            format_id = gdi32.ChoosePixelFormat(
                self._device_context, ctypes.byref(pixel_format)
            )
            if format_id == 0 or not gdi32.SetPixelFormat(
                self._device_context, format_id, ctypes.byref(pixel_format)
            ):
                self.free()
                raise win_error(get_last_error())
            self._context = opengl32.wglCreateContext(self._device_context)
            if not self._context:
                self.free()
                raise win_error(get_last_error())

        @classmethod
        def _ensure_window_class(cls) -> None:
            if cls._registered:
                return
            window_class = _WndClass()
            window_class.style = 0x0020  # CS_OWNDC
            window_class.lpfnWndProc = cls._window_proc
            window_class.hInstance = kernel32.GetModuleHandleW(None)
            window_class.lpszClassName = cls._class_name
            if not user32.RegisterClassW(ctypes.byref(window_class)):
                error = get_last_error()
                if error != 1410:  # ERROR_CLASS_ALREADY_EXISTS
                    raise win_error(error)
            cls._registered = True

        def make_current(self) -> None:
            if not opengl32.wglMakeCurrent(self._device_context, self._context):
                raise win_error(get_last_error())

        def free(self) -> None:
            if self._context:
                opengl32.wglMakeCurrent(None, None)
                opengl32.wglDeleteContext(self._context)
                self._context = None
            if self._window and self._device_context:
                user32.ReleaseDC(self._window, self._device_context)
                self._device_context = None
            if self._window:
                user32.DestroyWindow(self._window)
                self._window = None

        def __del__(self) -> None:
            self.free()

    gl_context.GLContext = _WglContext
    mujoco.GLContext = _WglContext
    mujoco.glfw.GLContext = _WglContext
    mujoco_glfw.GLContext = _WglContext
    classic_gl_context.GLContext = _WglContext
    classic_renderer.GLContext = _WglContext
    classic_renderer.gl_context.GLContext = _WglContext
    configure_windows_mujoco_renderer._configured = True  # type: ignore[attr-defined]
