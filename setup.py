#!/usr/bin/env python3

"""Package configuration for EnvPool."""

import os
import shutil
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.install import install
from setuptools.dist import Distribution


class InstallPlatlib(install):
    """Fix auditwheel error, https://github.com/google/or-tools/issues/616."""

    def finalize_options(self) -> None:
        """Install extension modules into the platform-specific directory."""
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib


class BinaryDistribution(Distribution):
    """Setuptools distribution that marks EnvPool as non-pure."""

    def is_pure(self) -> bool:
        """Return `False` because EnvPool ships extension modules."""
        return False

    def has_ext_modules(foo) -> bool:
        """Report that EnvPool has extension modules."""
        return True


class BuildPy(build_py):
    """Normalize Windows Bazel pybind outputs to importable `.pyd` files."""

    def run(self) -> None:
        """Rename copied Windows pybind artifacts to Python import suffixes."""
        super().run()
        if sys.platform != "win32":
            return
        self._rename_windows_extension_modules()
        self._copy_windows_procgen_runtime_dlls()

    def _rename_windows_extension_modules(self) -> None:
        """Strip Bazel's trailing `.dll` from copied Windows extension modules."""
        for dll_path in Path(self.build_lib).rglob("*.pyd.dll"):
            dll_path.rename(dll_path.with_suffix(""))

    def _copy_windows_procgen_runtime_dlls(self) -> None:
        procgen_dir = Path(self.build_lib) / "envpool" / "procgen"
        if not (procgen_dir / "procgen_envpool.pyd").is_file():
            return

        qt_root = os.environ.get("BAZEL_RULES_QT_DIR") or os.environ.get(
            "QT_ROOT_DIR"
        )
        if not qt_root:
            raise FileNotFoundError(
                "Windows procgen wheels require BAZEL_RULES_QT_DIR or "
                "QT_ROOT_DIR to locate Qt runtime DLLs."
            )

        qt_bin = Path(qt_root) / "bin"
        # Python 3.8+ no longer resolves extension DLL dependencies from PATH.
        for dll_name in ("Qt5Core.dll", "Qt5Gui.dll"):
            source = qt_bin / dll_name
            if not source.is_file():
                raise FileNotFoundError(
                    f"Missing required Qt runtime DLL for procgen: {source}"
                )
            shutil.copy2(source, procgen_dir / dll_name)


if __name__ == "__main__":
    setup(
        distclass=BinaryDistribution,
        cmdclass={"build_py": BuildPy, "install": InstallPlatlib},
    )
