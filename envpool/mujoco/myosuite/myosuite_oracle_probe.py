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
"""Pinned official MyoSuite oracle helper.

This binary is used only by tests. It intentionally runs in a separate Python
process from EnvPool so the official MyoSuite dependencies can stay pinned to
the upstream v2.11.6 contract without replacing EnvPool's normal runtime deps.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import importlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

# MyoSuite projects normalized muscle actions through np.exp(float32). NumPy's
# optional x86 SIMD kernels and its scalar kernel differ by single-ULP amounts,
# so pin the oracle helper to the portable baseline before NumPy is imported.
_NUMPY_X86_BASELINE_FEATURE_MASK = (
    "AVX",
    "AVX2",
    "FMA3",
    "F16C",
    "SSE42",
    "SSE41",
    "POPCNT",
    "SSSE3",
    "AVX512F",
    "AVX512CD",
    "AVX512_SKX",
    "AVX512_CLX",
    "AVX512_CNL",
    "AVX512_ICL",
    "AVX512_SPR",
)
if platform.machine().lower() in {"amd64", "x86_64"}:
    os.environ.setdefault(
        "NPY_DISABLE_CPU_FEATURES",
        ",".join(_NUMPY_X86_BASELINE_FEATURE_MASK),
    )

import numpy as np

from envpool.python.glfw_context import preload_windows_gl_dlls

if platform.system() == "Windows":
    preload_windows_gl_dlls(strict=True)


def _runfiles_root() -> Path:
    path = Path(__file__).absolute()
    for parent in (path, *path.parents):
        if parent.name.endswith(".runfiles"):
            return parent
    path = Path(__file__).resolve()
    runfiles_dir = os.environ.get("RUNFILES_DIR")
    if runfiles_dir:
        return Path(runfiles_dir)
    if "TEST_SRCDIR" in os.environ:
        return Path(os.environ["TEST_SRCDIR"])
    return path.parents[3]


def _runfiles_manifests(runfiles: Path) -> tuple[Path, ...]:
    manifests = []
    env_manifest = os.environ.get("RUNFILES_MANIFEST_FILE")
    if env_manifest:
        manifests.append(Path(env_manifest))
    manifests.extend([
        runfiles / "MANIFEST",
        runfiles.parent / f"{runfiles.name}_manifest",
    ])

    unique_manifests = []
    seen = set()
    for manifest in manifests:
        key = os.fspath(manifest)
        if key not in seen:
            unique_manifests.append(manifest)
            seen.add(key)
    return tuple(unique_manifests)


def _mujoco_shared_lib_name() -> str | None:
    system = platform.system()
    if system == "Darwin":
        return "libmujoco.3.6.0.dylib"
    if system == "Windows":
        return "mujoco.dll"
    return None


def _bazel_mujoco_shared_lib_path() -> Path:
    shared_lib = _mujoco_shared_lib_name()
    if shared_lib is None:
        raise RuntimeError(
            f"no Bazel-built MuJoCo shared library for {platform.system()}"
        )
    runfiles = _runfiles_root()
    workspace = os.environ.get("TEST_WORKSPACE", "envpool")
    manifest_keys = (
        f"mujoco/{shared_lib}",
        f"{workspace}/external/mujoco/{shared_lib}",
    )
    for manifest in _runfiles_manifests(runfiles):
        if not manifest.is_file():
            continue
        with manifest.open(encoding="utf-8") as f:
            for line in f:
                logical_path, _, real_path = line.rstrip("\n").partition(" ")
                if logical_path not in manifest_keys:
                    continue
                candidate = Path(real_path)
                if (
                    candidate.is_file()
                    and "site-packages" not in candidate.parts
                ):
                    return candidate

    candidates = (
        runfiles / "mujoco" / shared_lib,
        runfiles / workspace / "external" / "mujoco" / shared_lib,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    for candidate in runfiles.rglob(shared_lib):
        if candidate.is_file() and "site-packages" not in candidate.parts:
            return candidate
    raise RuntimeError(
        f"could not locate Bazel-built {shared_lib} under {runfiles}"
    )


def _configure_mujoco_package_shared_lib() -> None:
    """Make the pinned oracle import use EnvPool's Bazel-built MuJoCo lib.

    Linux uses the pinned pip MuJoCo wheel directly. Replacing or preloading the
    package library there corrupts the Python binding's model-name reads in
    MuJoCo 3.6.0, while the pip wheel already works with the EGL render path.
    """
    shared_lib = _mujoco_shared_lib_name()
    if shared_lib is None or getattr(
        _configure_mujoco_package_shared_lib, "_configured", False
    ):
        return

    spec = importlib.util.find_spec("mujoco")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("could not locate pinned mujoco Python package")
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    if not (package_dir / "__init__.py").is_file():
        raise RuntimeError(f"invalid mujoco package path: {package_dir}")

    patched_root = Path(tempfile.mkdtemp(prefix="mujoco-oracle-"))
    atexit.register(shutil.rmtree, patched_root, ignore_errors=True)
    patched_package = patched_root / "mujoco"
    shutil.copytree(package_dir, patched_package, symlinks=False)
    shutil.copy2(_bazel_mujoco_shared_lib_path(), patched_package / shared_lib)
    sys.path.insert(0, str(patched_root))
    _configure_mujoco_package_shared_lib._configured = True  # type: ignore[attr-defined]


def _configure_macos_mujoco_renderer() -> None:
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


def _configure_windows_mujoco_renderer() -> None:
    """Match EnvPool's native WGL context for official render-oracle tests."""
    if platform.system() != "Windows" or getattr(
        _configure_windows_mujoco_renderer, "_configured", False
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
        _fields_ = [
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
        _fields_ = [
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
    _configure_windows_mujoco_renderer._configured = True  # type: ignore[attr-defined]


def _configure_linux_mujoco_renderer(render: bool) -> None:
    """Force the pinned oracle onto EnvPool CI's headless EGL renderer."""
    if not render or platform.system() != "Linux":
        return

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")


def _link_or_copy_file(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _overlay_tree(
    source: Path,
    destination: Path,
    *,
    ignore: Any = None,
    prefer_directory_symlink: bool = True,
) -> None:
    if prefer_directory_symlink:
        try:
            os.symlink(source, destination, target_is_directory=True)
            return
        except OSError:
            pass
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        copy_function=_link_or_copy_file,
        ignore=ignore,
    )


def _oracle_source_path() -> Path:
    runfiles = _runfiles_root()
    source = runfiles / "myosuite_source/myosuite"
    if not (source / "__init__.py").is_file():
        raise RuntimeError(f"could not locate MyoSuite source at {source}")
    assembled = Path(tempfile.mkdtemp(prefix="myosuite-oracle-"))
    atexit.register(shutil.rmtree, assembled, ignore_errors=True)
    package = assembled / "myosuite"
    _overlay_tree(
        source,
        package,
        ignore=lambda _root, names: (
            {"simhive"} if "simhive" in names else set()
        ),
        prefer_directory_symlink=False,
    )
    simhive = package / "simhive"
    simhive.mkdir()
    for repo, name in (
        ("myosuite_mpl_sim", "MPL_sim"),
        ("myosuite_ycb_sim", "YCB_sim"),
        ("myosuite_furniture_sim", "furniture_sim"),
        ("myosuite_myo_sim", "myo_sim"),
        ("myosuite_object_sim", "object_sim"),
    ):
        repo_path = runfiles / repo
        if not repo_path.is_dir():
            raise RuntimeError(f"could not locate {repo_path}")
        _overlay_tree(repo_path, simhive / name)
    return assembled


def _import_official() -> tuple[Any, Any, Any]:
    warnings.filterwarnings("ignore")
    _configure_mujoco_package_shared_lib()
    sys.path.insert(0, str(_oracle_source_path()))
    _configure_macos_mujoco_renderer()
    _configure_windows_mujoco_renderer()
    official_myosuite = importlib.import_module("myosuite")
    gym = importlib.import_module("myosuite.utils").gym
    gym_registry_specs = official_myosuite.gym_registry_specs
    return official_myosuite, gym_registry_specs, gym


def _space_report(task_ids: list[str]) -> dict[str, Any]:
    official_myosuite, gym_registry_specs, gym = _import_official()
    registry = gym_registry_specs()
    tasks: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        spec = registry[task_id]
        env = gym.make(task_id)
        try:
            tasks[task_id] = {
                "action_shape": list(env.action_space.shape),
                "max_episode_steps": int(spec.max_episode_steps),
                "observation_shape": list(env.observation_space.shape),
            }
        except Exception as exc:
            raise RuntimeError(f"oracle space failed for {task_id}") from exc
        finally:
            env.close()
    return {
        "ids": list(official_myosuite.myosuite_env_suite),
        "tasks": tasks,
        "version": official_myosuite.__version__,
    }


def _array(value: Any) -> np.ndarray:
    return np.asarray(value)


def _jsonable_array(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable_array(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable_array(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    array = _array(value)
    if array.ndim == 0:
        return array.item()
    if array.dtype == object:
        return [str(item) for item in array.ravel()]
    return array.tolist()


def _names_from_ids(model: Any, obj_type: Any, ids: list[int]) -> list[str]:
    import mujoco

    raw_model = model.ptr if hasattr(model, "ptr") else model
    return [
        mujoco.mj_id2name(raw_model, int(obj_type), int(obj_id))
        for obj_id in ids
    ]


def _metadata_report(task_ids: list[str]) -> dict[str, Any]:
    official_myosuite, _, gym = _import_official()
    import mujoco

    tasks: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        env = gym.make(task_id)
        try:
            unwrapped = env.unwrapped
            model = unwrapped.sim.model
            data = unwrapped.sim.data
            task: dict[str, Any] = {
                "action_shape": list(env.action_space.shape),
                "entry_class": type(unwrapped).__name__,
                "frame_skip": int(unwrapped.frame_skip),
                "init_qpos": _jsonable_array(unwrapped.init_qpos),
                "init_qvel": _jsonable_array(unwrapped.init_qvel),
                "model_nq": int(model.nq),
                "model_nv": int(model.nv),
                "model_na": int(model.na),
                "model_nu": int(model.nu),
                "obs_keys": list(unwrapped.obs_keys),
                "observation_shape": list(env.observation_space.shape),
                "rwd_keys_wt": dict(unwrapped.rwd_keys_wt),
            }
            for attr in (
                "far_th",
                "goal_th",
                "hip_period",
                "max_rot",
                "min_height",
                "pose_thd",
                "reset_type",
                "target_rot",
                "target_x_vel",
                "target_y_vel",
                "terrain",
                "variant",
            ):
                if hasattr(unwrapped, attr):
                    task[attr] = _jsonable_array(getattr(unwrapped, attr))
            if hasattr(unwrapped, "tip_sids"):
                task["tip_sites"] = _names_from_ids(
                    model, mujoco.mjtObj.mjOBJ_SITE, unwrapped.tip_sids
                )
            if hasattr(unwrapped, "target_sids"):
                task["target_sites"] = _names_from_ids(
                    model, mujoco.mjtObj.mjOBJ_SITE, unwrapped.target_sids
                )
            if hasattr(unwrapped, "target_jnt_ids"):
                task["target_joints"] = _names_from_ids(
                    model, mujoco.mjtObj.mjOBJ_JOINT, unwrapped.target_jnt_ids
                )
            for attr in (
                "target_jnt_range",
                "target_jnt_value",
                "target_reach_range",
            ):
                if hasattr(unwrapped, attr):
                    task[attr] = _jsonable_array(getattr(unwrapped, attr))
            task["initial_state"] = {
                "qpos": _jsonable_array(data.qpos),
                "qvel": _jsonable_array(data.qvel),
                "act": _jsonable_array(data.act) if model.na > 0 else [],
                "qacc_warmstart": _jsonable_array(data.qacc_warmstart),
                "site_pos": _jsonable_array(model.site_pos),
                "site_quat": _jsonable_array(model.site_quat),
                "body_pos": _jsonable_array(model.body_pos),
                "body_quat": _jsonable_array(model.body_quat),
            }
            env.reset(seed=0)
            task["reset_state"] = _state_report(unwrapped)
            tasks[task_id] = task
        finally:
            env.close()
    return {"tasks": tasks, "version": official_myosuite.__version__}


def _state_report(env: Any) -> dict[str, Any]:
    model = env.sim.model
    data = env.sim.data
    state = {
        "act": _jsonable_array(data.act) if model.na > 0 else [],
        "actuator_force": _jsonable_array(data.actuator_force),
        "actuator_length": _jsonable_array(data.actuator_length),
        "actuator_velocity": _jsonable_array(data.actuator_velocity),
        "ctrl": _jsonable_array(data.ctrl),
        "geom_xpos": _jsonable_array(data.geom_xpos),
        "geom_xmat": _jsonable_array(data.geom_xmat),
        "geom_rgba": _jsonable_array(model.geom_rgba),
        "qacc_warmstart": _jsonable_array(data.qacc_warmstart),
        "body_pos": _jsonable_array(model.body_pos),
        "body_quat": _jsonable_array(model.body_quat),
        "light_xdir": _jsonable_array(data.light_xdir),
        "light_xpos": _jsonable_array(data.light_xpos),
        "mocap_pos": _jsonable_array(data.mocap_pos),
        "mocap_quat": _jsonable_array(data.mocap_quat),
        "qpos": _jsonable_array(data.qpos),
        "qvel": _jsonable_array(data.qvel),
        "site_pos": _jsonable_array(model.site_pos),
        "site_quat": _jsonable_array(model.site_quat),
        "site_size": _jsonable_array(model.site_size),
        "site_xpos": _jsonable_array(data.site_xpos),
        "site_rgba": _jsonable_array(model.site_rgba),
        "time": float(data.time),
    }
    fatigue = getattr(env, "muscle_fatigue", None)
    if fatigue is not None:
        state.update({
            "fatigue_ma": _jsonable_array(fatigue._MA),
            "fatigue_mr": _jsonable_array(fatigue._MR),
            "fatigue_mf": _jsonable_array(fatigue._MF),
            "fatigue_tl": _jsonable_array(fatigue.TL),
            "fatigue_tauact": _jsonable_array(fatigue._tauact),
            "fatigue_taudeact": _jsonable_array(fatigue._taudeact),
            "fatigue_dt": float(fatigue._dt),
        })
    return state


def _state_array(
    state: dict[str, Any], key: str, shape: tuple[int, ...]
) -> np.ndarray | None:
    value = state.get(key)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    size = int(np.prod(shape, dtype=np.int64))
    if array.size < size:
        raise ValueError(
            f"sync state {key} has {array.size} values, expected {size}"
        )
    return array[:size].reshape(shape)


def _assign_sync_array(
    state: dict[str, Any], key: str, target: np.ndarray
) -> None:
    value = state.get(key)
    if value is None:
        return
    array = np.asarray(value, dtype=np.float64).ravel()
    target_flat = target.reshape(-1)
    count = min(array.size, target_flat.size)
    target_flat[:count] = array[:count]
    if count < target_flat.size:
        target_flat[count:] = 0.0


def _assign_sync_array_if_same_size(
    state: dict[str, Any], key: str, target: np.ndarray
) -> None:
    value = state.get(key)
    if value is None:
        return
    array = np.asarray(value, dtype=np.float64).ravel()
    if array.size != target.size:
        return
    target.reshape(-1)[:] = array


def _sync_osl_phase_from_qpos(env: Any) -> None:
    controller = getattr(env, "OSL_CTRL", None)
    if controller is None:
        return
    model = env.sim.model
    data = env.sim.data
    if model.nkey < 3:
        controller.reset("e_stance")
        controller.start()
        return
    qpos = np.asarray(data.qpos, dtype=np.float64)
    key_qpos = np.asarray(model.key_qpos, dtype=np.float64).reshape(
        model.nkey, model.nq
    )
    start = min(7, model.nq)
    distances = np.sum((key_qpos[:3, start:] - qpos[start:]) ** 2, axis=1)
    phase = "e_swing" if int(np.argmin(distances)) == 1 else "e_stance"
    controller.reset(phase)
    controller.start()


def _sync_baoding_goal_from_envpool_reset_state(env: Any) -> None:
    if not all(
        hasattr(env, attr)
        for attr in (
            "ball_1_starting_angle",
            "ball_2_starting_angle",
            "center_pos",
            "create_goal_trajectory",
            "x_radius",
            "y_radius",
        )
    ):
        return
    task_type = type(getattr(env, "which_task", object()))
    if hasattr(task_type, "BAODING_CCW"):
        env.which_task = task_type.BAODING_CCW
    env.ball_1_starting_angle = np.pi / 4.0
    env.ball_2_starting_angle = env.ball_1_starting_angle - np.pi
    env.center_pos = np.array([-0.0125, -0.07], dtype=np.float64)
    env.x_radius = 0.025
    env.y_radius = 0.028
    env.goal = env.create_goal_trajectory(
        time_step=float(getattr(env, "dt", 0.025)), time_period=6.0
    )
    env.counter = 0


def _sync_chasetag_hidden_state(env: Any) -> None:
    if not all(hasattr(env, attr) for attr in ("current_task", "opponent")):
        return
    task_type = type(env.current_task)
    if hasattr(task_type, "CHASE"):
        env.current_task = task_type.CHASE
    opponent = env.opponent
    opponent.opponent_policy = "stationary"
    opponent.opponent_vel = np.zeros((2,), dtype=np.float64)
    if hasattr(opponent, "chase_velocity"):
        opponent.chase_velocity = 1.0


def _sync_fatigue_hidden_state(env: Any, state: dict[str, Any]) -> None:
    fatigue = getattr(env, "muscle_fatigue", None)
    if fatigue is None:
        return
    _assign_sync_array(state, "fatigue_ma", fatigue._MA)
    _assign_sync_array(state, "fatigue_mr", fatigue._MR)
    _assign_sync_array(state, "fatigue_mf", fatigue._MF)
    _assign_sync_array(state, "fatigue_tl", fatigue.TL)


def _sync_to_envpool_reset_state(env: Any, state: dict[str, Any]) -> np.ndarray:
    """Patch the official oracle to EnvPool's reset-time MuJoCo state once."""
    sim = env.sim
    model = sim.model
    data = sim.data

    _assign_sync_array(state, "site_pos", model.site_pos)
    _assign_sync_array(state, "site_quat", model.site_quat)
    _assign_sync_array(state, "site_size", model.site_size)
    _assign_sync_array(state, "site_rgba", model.site_rgba)
    _assign_sync_array(state, "body_pos", model.body_pos)
    _assign_sync_array(state, "body_quat", model.body_quat)
    _assign_sync_array(state, "body_mass", model.body_mass)
    _assign_sync_array(state, "geom_pos", model.geom_pos)
    _assign_sync_array(state, "geom_quat", model.geom_quat)
    _assign_sync_array(state, "geom_size", model.geom_size)
    _assign_sync_array(state, "geom_rgba", model.geom_rgba)
    _assign_sync_array(state, "geom_friction", model.geom_friction)
    _assign_sync_array_if_same_size(state, "geom_aabb", model.geom_aabb)
    _assign_sync_array_if_same_size(state, "geom_rbound", model.geom_rbound)
    _assign_sync_array_if_same_size(state, "geom_contype", model.geom_contype)
    _assign_sync_array_if_same_size(
        state, "geom_conaffinity", model.geom_conaffinity
    )
    _assign_sync_array_if_same_size(state, "geom_type", model.geom_type)
    _assign_sync_array_if_same_size(state, "geom_condim", model.geom_condim)
    _assign_sync_array(state, "hfield_data", model.hfield_data)
    if model.nmocap > 0:
        _assign_sync_array(state, "mocap_pos", data.mocap_pos)
        _assign_sync_array(state, "mocap_quat", data.mocap_quat)

    qpos = _state_array(state, "qpos0", data.qpos.shape)
    qvel = _state_array(state, "qvel0", data.qvel.shape)
    act = _state_array(state, "act0", data.act.shape) if model.na > 0 else None
    sim.set_state(time=0.0, qpos=qpos, qvel=qvel, act=act)

    _assign_sync_array(state, "ctrl", data.ctrl)
    sim.forward()
    _sync_osl_phase_from_qpos(env)
    _sync_baoding_goal_from_envpool_reset_state(env)
    _sync_chasetag_hidden_state(env)
    _sync_fatigue_hidden_state(env, state)
    obs = env.get_obs()
    _assign_sync_array(state, "qacc0", data.qacc)
    _assign_sync_array(state, "qacc_warmstart0", data.qacc_warmstart)
    if hasattr(env, "last_ctrl"):
        env.last_ctrl = data.ctrl.copy()
    return obs


def _trace_info(info: dict[str, Any]) -> dict[str, Any]:
    scalar_info: dict[str, Any] = {}
    for key in ("rwd_dense", "rwd_sparse", "solved", "done", "time"):
        if key in info:
            scalar_info[key] = _jsonable_array(info[key])
    if "rwd_dict" in info:
        scalar_info["rwd_dict"] = {
            key: _jsonable_array(value)
            for key, value in info["rwd_dict"].items()
            if np.asarray(value).size <= 16
        }
    return scalar_info


def _needs_cgl_warmup_render(task_id: str) -> bool:
    return (
        "Challenge" in task_id
        or "Elbow" in task_id
        or task_id
        in {
            "motorFingerReachFixed-v0",
            "motorFingerReachRandom-v0",
            "myoFingerPoseFixed-v0",
            "myoFingerPoseRandom-v0",
            "myoFingerReachFixed-v0",
            "myoFingerReachRandom-v0",
        }
    )


def _render_frame(
    task_id: str, env: Any, width: int, height: int, camera_id: int
) -> Any:
    env.unwrapped.sim.forward()
    renderer = env.unwrapped.sim.renderer
    frame = renderer.render_offscreen(
        width=width,
        height=height,
        camera_id=camera_id,
    )
    if (
        platform.system() == "Darwin"
        and _needs_cgl_warmup_render(task_id)
        and not getattr(renderer, "_envpool_cgl_first_render_done", False)
    ):
        renderer._envpool_cgl_first_render_done = True
        for _ in range(32):
            frame = renderer.render_offscreen(
                width=width,
                height=height,
                camera_id=camera_id,
            )
    return frame


def _next_action(
    rng: np.random.Generator,
    low: np.ndarray,
    high: np.ndarray,
    action_mode: str,
) -> np.ndarray:
    if action_mode == "random":
        return rng.uniform(low, high).astype(np.float32)
    if action_mode == "midpoint":
        return ((low + high) * 0.5).astype(np.float32)
    if action_mode == "zero":
        return np.clip(np.zeros_like(low), low, high).astype(np.float32)
    raise ValueError(f"unknown action mode: {action_mode}")


def _rollout_report(
    task_ids: list[str], steps: int, seed: int, action_mode: str
) -> dict[str, Any]:
    official_myosuite, _, gym = _import_official()
    rng = np.random.default_rng(seed + 17)
    tasks: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        env = gym.make(task_id)
        try:
            reset = env.reset(seed=seed)
            obs = reset[0] if isinstance(reset, tuple) else reset
            low = _array(env.action_space.low).astype(np.float32)
            high = _array(env.action_space.high).astype(np.float32)
            rewards: list[float] = []
            terminals: list[bool] = []
            truncateds: list[bool] = []
            obs_checksum = [float(_array(obs).astype(np.float64).sum())]
            for _ in range(steps):
                action = _next_action(rng, low, high, action_mode)
                step = env.step(action)
                obs = step[0]
                rewards.append(float(step[1]))
                terminals.append(bool(step[2]))
                truncateds.append(bool(step[3]) if len(step) > 4 else False)
                obs_checksum.append(float(_array(obs).astype(np.float64).sum()))
            tasks[task_id] = {
                "obs_checksum": obs_checksum,
                "rewards": rewards,
                "terminated": terminals,
                "truncated": truncateds,
            }
        finally:
            env.close()
    return {"tasks": tasks, "version": official_myosuite.__version__}


def _trace_report(
    task_ids: list[str],
    steps: int,
    seed: int,
    render: bool,
    render_width: int,
    render_height: int,
    camera_id: int,
    action_mode: str,
    sync_states: dict[str, Any] | None = None,
    trace_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    official_myosuite, _, gym = _import_official()
    rng = np.random.default_rng(seed + 17)
    tasks: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        task_plan = trace_plan.get(task_id, {}) if trace_plan else {}
        planned_actions = task_plan.get("actions")
        planned_resets = task_plan.get("reset_before_step", [])
        planned_sync_states = task_plan.get("sync_states", [])
        env = gym.make(task_id)
        try:
            reset = env.reset(seed=seed)
            obs = reset[0] if isinstance(reset, tuple) else reset
            unwrapped = env.unwrapped
            if planned_sync_states:
                obs = _sync_to_envpool_reset_state(
                    unwrapped, planned_sync_states[0]
                )
            elif sync_states is not None and task_id in sync_states:
                obs = _sync_to_envpool_reset_state(
                    unwrapped, sync_states[task_id]
                )
            low = _array(env.action_space.low).astype(np.float32)
            high = _array(env.action_space.high).astype(np.float32)
            frames: list[Any] = []
            if render:
                frames.append(
                    _jsonable_array(
                        _render_frame(
                            task_id,
                            env,
                            render_width,
                            render_height,
                            camera_id,
                        )
                    )
                )
            trace: dict[str, Any] = {
                "actions": [],
                "infos": [],
                "obs": [_jsonable_array(obs)],
                "reset_state": _state_report(unwrapped),
                "rewards": [],
                "states": [],
                "terminated": [],
                "truncated": [],
            }
            trace_steps = (
                len(planned_actions) if planned_actions is not None else steps
            )
            for step_id in range(trace_steps):
                if planned_actions is None:
                    action = _next_action(rng, low, high, action_mode)
                else:
                    action = np.asarray(
                        planned_actions[step_id], dtype=np.float32
                    )
                reset_before_step = step_id < len(planned_resets) and bool(
                    planned_resets[step_id]
                )
                trace["actions"].append(_jsonable_array(action))
                if reset_before_step:
                    reset = env.reset()
                    obs = reset[0] if isinstance(reset, tuple) else reset
                    if step_id + 1 < len(planned_sync_states):
                        obs = _sync_to_envpool_reset_state(
                            unwrapped, planned_sync_states[step_id + 1]
                        )
                    else:
                        env.sim.forward()
                    trace["obs"].append(_jsonable_array(obs))
                    trace["rewards"].append(0.0)
                    trace["terminated"].append(False)
                    trace["truncated"].append(False)
                    trace["infos"].append({})
                else:
                    step = env.step(action)
                    obs = step[0]
                    trace["obs"].append(_jsonable_array(obs))
                    trace["rewards"].append(float(step[1]))
                    trace["terminated"].append(bool(step[2]))
                    trace["truncated"].append(
                        bool(step[3]) if len(step) > 4 else False
                    )
                    trace["infos"].append(_trace_info(step[-1]))
                state = _state_report(unwrapped)
                if hasattr(unwrapped, "last_ctrl"):
                    state["last_ctrl"] = _jsonable_array(unwrapped.last_ctrl)
                trace["states"].append(state)
                if render:
                    frames.append(
                        _jsonable_array(
                            _render_frame(
                                task_id,
                                env,
                                render_width,
                                render_height,
                                camera_id,
                            )
                        )
                    )
            if render:
                trace["frames"] = frames
            tasks[task_id] = trace
        except Exception as exc:
            raise RuntimeError(f"oracle trace failed for {task_id}") from exc
        finally:
            env.close()
    return {"tasks": tasks, "version": official_myosuite.__version__}


def main() -> None:
    """Run the requested pinned-oracle probe and write a JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("metadata", "space", "rollout", "trace"),
        required=True,
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_width", type=int, default=64)
    parser.add_argument("--render_height", type=int, default=48)
    parser.add_argument("--camera_id", type=int, default=-1)
    parser.add_argument("--sync_state")
    parser.add_argument("--trace_plan")
    parser.add_argument(
        "--action_mode",
        choices=("random", "midpoint", "zero"),
        default="random",
    )
    parser.add_argument("--task_id", action="append", default=[])
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    _configure_linux_mujoco_renderer(args.render)

    sync_states = (
        json.loads(Path(args.sync_state).read_text())
        if args.sync_state is not None
        else None
    )
    trace_plan = (
        json.loads(Path(args.trace_plan).read_text())
        if args.trace_plan is not None
        else None
    )

    if args.mode == "space":
        report = _space_report(args.task_id)
    elif args.mode == "rollout":
        report = _rollout_report(
            args.task_id, args.steps, args.seed, args.action_mode
        )
    elif args.mode == "trace":
        report = _trace_report(
            args.task_id,
            args.steps,
            args.seed,
            args.render,
            args.render_width,
            args.render_height,
            args.camera_id,
            args.action_mode,
            sync_states,
            trace_plan,
        )
    else:
        report = _metadata_report(args.task_id)
    Path(args.out).write_text(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
