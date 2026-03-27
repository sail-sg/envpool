#!/usr/bin/env python3

"""Package configuration for EnvPool."""

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
        for dll_path in Path(self.build_lib).rglob("*_envpool.pyd.dll"):
            dll_path.rename(dll_path.with_suffix(""))


if __name__ == "__main__":
    setup(
        distclass=BinaryDistribution,
        cmdclass={"build_py": BuildPy, "install": InstallPlatlib},
    )
