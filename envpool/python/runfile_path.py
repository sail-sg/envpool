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

"""Locate Bazel runfiles for tests and documentation tools."""

from functools import cache
from pathlib import Path

from python.runfiles import runfiles


@cache
def _runfiles() -> runfiles.Runfiles:
    files = runfiles.Create()
    if files is None:
        raise RuntimeError("Run this tool with Bazel to locate its data files")
    return files


def find_runfile(path: str) -> Path:
    """Resolve a file path starting with its apparent Bazel repository."""
    location = _runfiles().Rlocation(path, source_repo="")
    if location is not None:
        candidate = Path(location)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find runfile {path!r}")
