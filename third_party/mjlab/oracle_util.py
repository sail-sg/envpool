# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cache bootstrap for the build-time exporter and isolated official oracle."""

import os
import sys
from pathlib import Path


def configure_cache(path: Path) -> None:
    """Use explicit caches and compiler-compatible physical paths in Bazel."""
    # Bazel's Windows action environment has no profile directory. MJLab
    # imports Parso/IPython through mediapy, even without a viewer or notebook.
    if sys.platform == "win32":
        os.environ.setdefault("USERPROFILE", str(path))
        os.environ.setdefault("LOCALAPPDATA", str(path))
        # Prefer Bazel's shorter physical wheel paths. Extended paths let
        # Python load deeply vendored W&B modules, but Warp's embedded Clang
        # cannot resolve nested includes through a \\?\ import path.
        wheel_paths = {}
        manifest = os.environ.get("RUNFILES_MANIFEST_FILE")
        if not manifest:
            for variable in ("RUNFILES_DIR", "TEST_SRCDIR"):
                if root := os.environ.get(variable):
                    candidate = Path(root) / "MANIFEST"
                    if candidate.is_file():
                        manifest = str(candidate)
                        break
        if manifest:
            for line in Path(manifest).read_text(encoding="utf-8").splitlines():
                logical, _, physical = line.partition(" ")
                prefix, separator, suffix = logical.partition("/site-packages/")
                normalized = physical.replace("\\", "/")
                if separator and normalized.endswith("/" + suffix):
                    wheel_paths[prefix + "/site-packages"] = normalized[
                        : -len(suffix) - 1
                    ]

        def extended(value: str) -> str:
            absolute = os.path.abspath(value)
            normalized = absolute.replace("\\", "/")
            for logical, physical in wheel_paths.items():
                if normalized.endswith("/" + logical):
                    return physical
            if absolute.startswith("\\\\?\\"):
                return absolute
            if absolute.startswith("\\\\"):
                return "\\\\?\\UNC\\" + absolute[2:]
            return "\\\\?\\" + absolute

        sys.path[:] = [extended(value) for value in sys.path]
    os.environ.setdefault("MPLCONFIGDIR", str(path / "matplotlib"))
