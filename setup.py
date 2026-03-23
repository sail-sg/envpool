#!/usr/bin/env python3

"""Package configuration for EnvPool."""

from setuptools import setup
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


if __name__ == "__main__":
    setup(distclass=BinaryDistribution, cmdclass={"install": InstallPlatlib})
