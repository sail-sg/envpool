#!/usr/bin/env python3

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):

  def has_ext_modules(foo) -> bool:
    return True


if __name__ == '__main__':
  setup(distclass=BinaryDistribution)
