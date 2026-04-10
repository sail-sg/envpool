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
"""Test helpers for importing the upstream highway-env oracle."""

from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path


def register_highway_envs() -> None:
    """Register EnvPool Highway tasks for tests.

    Tests import make_gymnasium from the low-level registry so Bazel can keep
    small runfiles; importing the registration module installs the tasks.
    """
    importlib.import_module("envpool.highway.registration")


def prepare_official_oracle_import() -> None:
    """Provide home/cache dirs before highway-env imports matplotlib.

    The upstream package imports matplotlib at module import time. Under the
    Windows Bazel test runner Python may not have an expandable home directory,
    which makes matplotlib's default config-dir lookup fail before tests start.
    """
    test_tmpdir = Path(os.environ.get("TEST_TMPDIR", tempfile.gettempdir()))
    home = test_tmpdir / "home"
    matplotlib_config = test_tmpdir / "matplotlib"
    home.mkdir(parents=True, exist_ok=True)
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HOME", str(home))
    os.environ.setdefault("USERPROFILE", str(home))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
