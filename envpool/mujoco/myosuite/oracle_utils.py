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
"""Shared MyoSuite oracle helpers for tests and docs generation."""

from __future__ import annotations

import ctypes
import importlib
import os
import platform
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any, Iterator

from envpool.mujoco.myosuite.paths import (
    myosuite_asset_root,
    resolve_workspace_path,
)


def _configure_linux_mujoco_gl() -> None:
    if platform.system() != "Linux":
        return
    if os.environ.get("MUJOCO_GL"):
        if os.environ["MUJOCO_GL"] == "egl":
            os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        return
    for backend in ("egl", "osmesa"):
        env = dict(os.environ)
        env["MUJOCO_GL"] = backend
        if backend == "egl":
            env.setdefault("EGL_PLATFORM", "surfaceless")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import mujoco; "
                    "ctx = mujoco.GLContext(1, 1); "
                    "ctx.make_current(); "
                    "ctx.free()"
                ),
            ],
            env=env,
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            os.environ["MUJOCO_GL"] = backend
            if backend == "egl":
                os.environ.setdefault("EGL_PLATFORM", "surfaceless")
            return


def _configure_macos_dm_control_imports() -> None:
    if platform.system() != "Darwin":
        return
    # dm_control eagerly imports the GLFW backend when MUJOCO_GL is unset.
    # Force the no-render backend at import time so non-render oracle tests do
    # not hang inside glfw.init() on macOS.
    os.environ.setdefault("MUJOCO_GL", "off")


def _configure_macos_dm_control_renderer() -> None:
    if platform.system() != "Darwin":
        return

    from dm_control import _render
    from dm_control._render import base as dm_control_render_base
    from dm_control._render import executor as dm_control_render_executor

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
                attrib.CGLPFAMultisample,
                attrib.CGLPFASampleBuffers,
                1,
                attrib.CGLPFASample,
                4,
                attrib.CGLPFAAccelerated,
                0,
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


def _configure_macos_mujoco_renderer() -> None:
    if platform.system() != "Darwin":
        return

    import mujoco.rendering.classic.gl_context as classic_gl_context
    import mujoco.rendering.classic.renderer as classic_renderer

    class _ClassicCglContext:
        def __init__(self, width: int, height: int):
            del width, height
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
                attrib.CGLPFAMultisample,
                attrib.CGLPFASampleBuffers,
                1,
                attrib.CGLPFASample,
                4,
                attrib.CGLPFAAccelerated,
                0,
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

        def make_current(self) -> None:
            from mujoco.cgl import cgl

            cgl.CGLSetCurrentContext(self._context)
            if not self._locked:
                cgl.CGLLockContext(self._context)
                self._locked = True

        def free(self) -> None:
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

        def __del__(self) -> None:
            self.free()

    os.environ["MUJOCO_GL"] = "cgl"
    classic_gl_context.GLContext = _ClassicCglContext
    classic_renderer.GLContext = _ClassicCglContext


def _replace_all(text: str, old: str, new: str) -> str:
    return text.replace(old, new)


def _relative_model_path(path: Path, *, oracle_dir: Path) -> str:
    prefix = "\\" if os.name == "nt" else "/"
    return prefix + os.path.relpath(path, oracle_dir)


def find_vendored_myosuite_root() -> Path:
    """Locate the vendored upstream MyoSuite Python source tree."""
    root = resolve_workspace_path(".")
    for candidate in (root, *root.parents):
        direct = candidate / "myosuite_src"
        if (direct / "myosuite/envs/myo/myobase/pose_v0.py").exists():
            return direct
        for pose_path in candidate.rglob(
            "myosuite/envs/myo/myobase/pose_v0.py"
        ):
            return pose_path.parents[4]
    raise FileNotFoundError("Unable to locate vendored myosuite source root")


@cache
def prepare_oracle_imports(*, render: bool = False) -> None:
    """Expose vendored MyoSuite Python modules on sys.path."""
    _configure_linux_mujoco_gl()
    _configure_macos_dm_control_imports()
    source_root = str(find_vendored_myosuite_root())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    if render:
        _configure_macos_dm_control_renderer()
        _configure_macos_mujoco_renderer()


@cache
def load_oracle_class(entry_module: str, class_name: str) -> Any:
    """Import one official MyoSuite environment class from vendored source."""
    prepare_oracle_imports()
    module = importlib.import_module(entry_module)
    return getattr(module, class_name)


@cache
def load_oracle_attr(module_path: str, attr_name: str) -> Any:
    """Import one vendored upstream attribute by module path and name."""
    prepare_oracle_imports()
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


@contextmanager
def prepared_track_oracle_model_path() -> Iterator[str]:
    """Yield a TrackEnv model_path that resolves to writable staged assets."""
    asset_root = myosuite_asset_root()
    source_model = asset_root / "envs/myo/assets/hand/myohand_object.xml"
    object_xml = source_model.read_text()
    tabletop_xml = (
        asset_root / "envs/myo/assets/hand/myohand_tabletop.xml"
    ).read_text()
    hand_assets_xml = (
        asset_root / "simhive/myo_sim/hand/assets/myohand_assets.xml"
    ).read_text()
    myo_sim_root = asset_root / "simhive/myo_sim"
    myo_sim_root_str = str(myo_sim_root)
    object_sim_root = asset_root / "simhive/object_sim"
    object_sim_root_str = str(object_sim_root)
    oracle_dir = find_vendored_myosuite_root() / "myosuite/envs/myo/myodm"

    with tempfile.TemporaryDirectory(prefix="envpool_myodm_oracle_") as td:
        tmp_dir = Path(td)
        hand_assets_tmp = tmp_dir / "myohand_assets.xml"
        tabletop_tmp = tmp_dir / "myohand_tabletop.xml"
        object_tmp = tmp_dir / "myohand_object.xml"

        hand_assets_xml = _replace_all(
            hand_assets_xml,
            'meshdir=".." texturedir=".."',
            f'meshdir="{myo_sim_root_str}" texturedir="{myo_sim_root_str}"',
        )
        hand_assets_tmp.write_text(hand_assets_xml)

        tabletop_xml = _replace_all(
            tabletop_xml,
            "../../../../simhive/myo_sim/hand/assets/myohand_assets.xml",
            str(hand_assets_tmp),
        )
        tabletop_xml = _replace_all(
            tabletop_xml,
            "../../../../simhive/furniture_sim/simpleTable/simpleTable_asset.xml",
            str(
                asset_root
                / "simhive/furniture_sim/simpleTable/simpleTable_asset.xml"
            ),
        )
        tabletop_xml = _replace_all(
            tabletop_xml,
            "../../../../simhive/myo_sim/hand/assets/myohand_body.xml",
            str(asset_root / "simhive/myo_sim/hand/assets/myohand_body.xml"),
        )
        tabletop_xml = _replace_all(
            tabletop_xml,
            "../../../../simhive/furniture_sim/simpleTable/"
            "simpleGraniteTable_body.xml",
            str(
                asset_root
                / "simhive/furniture_sim/simpleTable/simpleGraniteTable_body.xml"
            ),
        )
        tabletop_xml = _replace_all(
            tabletop_xml,
            'meshdir="../../../../simhive/myo_sim/" texturedir="../../../../simhive/myo_sim/"',
            f'meshdir="{myo_sim_root_str}" texturedir="{myo_sim_root_str}"',
        )
        tabletop_tmp.write_text(tabletop_xml)

        object_xml = _replace_all(
            object_xml, "myohand_tabletop.xml", str(tabletop_tmp)
        )
        object_xml = _replace_all(
            object_xml,
            "../../../../simhive/object_sim/common.xml",
            str(object_sim_root / "common.xml"),
        )
        object_xml = _replace_all(
            object_xml,
            "../../../../simhive/object_sim/OBJECT_NAME/assets.xml",
            object_sim_root_str + "/OBJECT_NAME/assets.xml",
        )
        object_xml = _replace_all(
            object_xml,
            "../../../../simhive/object_sim/OBJECT_NAME/body.xml",
            object_sim_root_str + "/OBJECT_NAME/body.xml",
        )
        object_tmp.write_text(object_xml)

        yield _relative_model_path(object_tmp, oracle_dir=oracle_dir)
