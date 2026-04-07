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

QT_HINT = "EnvPool Procgen requires the system Qt 5 runtime on Linux."


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
        choices=("present", "absent"),
        required=True,
        help="Whether the platform Qt runtime is expected to be installed.",
    )
    args = parser.parse_args()

    if args.qt_runtime == "present":
        _present_smoke()
    else:
        if not sys.platform.startswith("linux"):
            raise SystemExit("--qt-runtime=absent is only valid on Linux")
        _absent_smoke()


if __name__ == "__main__":
    main()
