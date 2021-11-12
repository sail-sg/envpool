#!/usr/bin/env python3

from setuptools import setup
from setuptools.command.install import install
from setuptools.dist import Distribution


class InstallPlatlib(install):
  """Fix auditwheel error, https://github.com/google/or-tools/issues/616"""

  def finalize_options(self) -> None:
    install.finalize_options(self)
    if self.distribution.has_ext_modules():
      self.install_lib = self.install_platlib


class BinaryDistribution(Distribution):

  def is_pure(self) -> bool:
    return False

  def has_ext_modules(foo) -> bool:
    return True


if __name__ == '__main__':
  setup(distclass=BinaryDistribution, cmdclass={'install': InstallPlatlib})
