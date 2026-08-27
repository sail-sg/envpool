#!/usr/bin/env python3
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
"""Release smoke tests for Procgen's Linux Qt runtime behavior."""

import argparse
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

QT_HINT = "EnvPool Procgen requires the system Qt runtime on Linux."


def _run_child(name: str, code: str) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="envpool-procgen-qt-") as tmpdir:
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            cwd=tmpdir,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    if result.returncode == 0:
        print(f"{name}: ok")
        print(result.stdout, end="")
        print(result.stderr, file=sys.stderr, end="")
        return
    print(f"{name}: failed", file=sys.stderr)
    print(result.stdout, file=sys.stderr, end="")
    print(result.stderr, file=sys.stderr, end="")
    raise SystemExit(result.returncode)


def _present_smoke() -> None:
    _run_child(
        "qt-present without explicit procgen import",
        """
        import envpool

        assert "CoinrunEasy-v0" in envpool.list_all_envs()
        env = envpool.make("Pong-v5", env_type="gymnasium", num_envs=1)
        try:
            env.reset()
        finally:
            env.close()
        """,
    )
    _run_child(
        "qt-present with explicit procgen import",
        """
        import envpool
        import envpool.procgen
        from envpool.procgen import ProcgenEnvSpec

        assert ProcgenEnvSpec is not None
        env = envpool.make("CoinrunEasy-v0", env_type="gymnasium", num_envs=1)
        try:
            env.reset()
        finally:
            env.close()
        """,
    )
    # Reuse the native consistency/pixel and all-task render suites against the
    # installed wheel. The optional upstream oracle is not a release dependency.
    for filename, tests in (
        (
            "procgen_test.py",
            [
                "_ProcgenEnvPoolTest.test_align",
                "_ProcgenEnvPoolTest.test_channel_first",
                "_ProcgenEnvPoolTest.test_deterministic",
            ],
        ),
        ("procgen_render_test.py", []),
    ):
        test_path = (
            Path(__file__).resolve().parents[1] / "envpool/procgen" / filename
        )
        _run_child(
            f"qt-present {filename}",
            f"""
            import runpy
            import sys

            sys.argv = {[str(test_path), *tests]!r}
            runpy.run_path({str(test_path)!r}, run_name="__main__")
            """,
        )


def _bundled_smoke() -> None:
    _run_child(
        "qt-bundled library origins",
        """
        from pathlib import Path
        import envpool
        from envpool.procgen.procgen_envpool import _qt_version

        assert _qt_version.startswith("6."), _qt_version
        library_dir = Path(envpool.__file__).resolve().parent.parent / "envpool.libs"
        loaded = {
            Path(line.split()[-1]).resolve()
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/libQt" in line
        }
        for module in ("Core", "Gui"):
            paths = {p for p in loaded if p.name.startswith(f"libQt6{module}")}
            assert paths, f"Qt6{module} was not loaded: {loaded}"
            assert all(p.parent == library_dir for p in paths), paths
        print(f"Qt {_qt_version} loaded from {library_dir}")
        """,
    )


def _absent_smoke() -> None:
    _run_child(
        "qt-absent without explicit procgen import",
        f"""
        import envpool

        assert "CoinrunEasy-v0" in envpool.list_all_envs()
        env = envpool.make("Pong-v5", env_type="gymnasium", num_envs=1)
        try:
            env.reset()
        finally:
            env.close()

        try:
            envpool.make("CoinrunEasy-v0", env_type="gymnasium", num_envs=1)
        except ImportError as exc:
            message = str(exc)
            assert {QT_HINT!r} in message, message
        else:
            raise AssertionError("Procgen unexpectedly worked without Qt")
        """,
    )
    _run_child(
        "qt-absent with explicit procgen import",
        f"""
        import envpool
        import envpool.procgen

        try:
            from envpool.procgen import ProcgenEnvSpec  # noqa: F401
        except ImportError as exc:
            message = str(exc)
            assert {QT_HINT!r} in message, message
        else:
            raise AssertionError("ProcgenEnvSpec unexpectedly imported without Qt")
        """,
    )


def main() -> None:
    """Run the selected Procgen Qt release smoke test."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qt-runtime",
        choices=("present", "absent", "bundled"),
        required=True,
        help="Expected Qt runtime; bundled also checks Linux library origins.",
    )
    args = parser.parse_args()

    if args.qt_runtime in ("present", "bundled"):
        if args.qt_runtime == "bundled" and not sys.platform.startswith(
            "linux"
        ):
            raise SystemExit("--qt-runtime=bundled is only valid on Linux")
        _present_smoke()
        if args.qt_runtime == "bundled":
            _bundled_smoke()
    else:
        if not sys.platform.startswith("linux"):
            raise SystemExit("--qt-runtime=absent is only valid on Linux")
        _absent_smoke()


if __name__ == "__main__":
    main()
