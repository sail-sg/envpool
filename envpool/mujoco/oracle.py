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
"""Share EnvPool's exact MuJoCo build with official test-only oracles."""

from __future__ import annotations

import atexit
import importlib.util
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path


def runfiles_root() -> Path:
    """Return the current Bazel runfiles tree."""
    path = Path(__file__).absolute()
    for parent in (path, *path.parents):
        if parent.name.endswith(".runfiles"):
            return parent
    if runfiles_dir := os.environ.get("RUNFILES_DIR"):
        return Path(runfiles_dir)
    if test_srcdir := os.environ.get("TEST_SRCDIR"):
        return Path(test_srcdir)
    return Path(__file__).resolve().parents[2]


def runfiles_repository(name: str) -> Path:
    """Resolve an apparent repository through Bazel's runfiles mapping."""
    root = runfiles_root()
    mapping = root / "_repo_mapping"
    if mapping.is_file():
        for entry in mapping.read_text(encoding="utf-8").splitlines():
            source, apparent, canonical = entry.split(",", 2)
            if not source and apparent == name:
                return root / canonical
    return root / name


def _runfiles_manifests(runfiles: Path) -> tuple[Path, ...]:
    candidates = []
    if manifest := os.environ.get("RUNFILES_MANIFEST_FILE"):
        candidates.append(Path(manifest))
    candidates.extend([
        runfiles / "MANIFEST",
        runfiles.parent / f"{runfiles.name}_manifest",
    ])
    return tuple(dict.fromkeys(candidates))


def _bazel_shared_library(name: str) -> Path:
    runfiles = runfiles_root()
    workspace = os.environ.get("TEST_WORKSPACE", "envpool")
    logical_paths = {
        f"mujoco/{name}",
        f"{workspace}/external/mujoco/{name}",
    }
    for manifest in _runfiles_manifests(runfiles):
        if not manifest.is_file():
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            logical_path, _, real_path = line.partition(" ")
            if logical_path in logical_paths:
                candidate = Path(real_path)
                if (
                    candidate.is_file()
                    and "site-packages" not in candidate.parts
                ):
                    return candidate

    for candidate in (
        runfiles / "mujoco" / name,
        runfiles / workspace / "external" / "mujoco" / name,
    ):
        if candidate.is_file():
            return candidate
    for candidate in runfiles.rglob(name):
        if candidate.is_file() and "site-packages" not in candidate.parts:
            return candidate
    raise RuntimeError(f"could not locate Bazel-built {name} under {runfiles}")


def configure_mujoco_package_shared_lib() -> None:
    """Use the identical source-built engine in macOS/Windows official oracles.

    Linux must keep its pinned pip wheel: replacing that package library
    corrupts Python binding model-name reads in MuJoCo 3.11.
    """
    system = platform.system()
    if system not in {"Darwin", "Windows"} or getattr(
        configure_mujoco_package_shared_lib, "_configured", False
    ):
        return

    spec = importlib.util.find_spec("mujoco")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("could not locate pinned mujoco Python package")
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    if not (package_dir / "__init__.py").is_file():
        raise RuntimeError(f"invalid mujoco package path: {package_dir}")

    if system == "Darwin":
        dylibs = tuple(package_dir.glob("libmujoco.*.dylib"))
        if len(dylibs) != 1:
            raise RuntimeError(f"expected one MuJoCo dylib under {package_dir}")
        library_name = dylibs[0].name
    else:
        library_name = "mujoco.dll"

    patched_root = Path(tempfile.mkdtemp(prefix="mujoco-oracle-"))
    atexit.register(shutil.rmtree, patched_root, ignore_errors=True)
    patched_package = patched_root / "mujoco"
    shutil.copytree(package_dir, patched_package, symlinks=False)
    shutil.copy2(
        _bazel_shared_library(library_name), patched_package / library_name
    )
    sys.path.insert(0, str(patched_root))
    configure_mujoco_package_shared_lib._configured = True  # type: ignore[attr-defined]
