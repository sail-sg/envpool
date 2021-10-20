from collections import OrderedDict
from copy import copy
from itertools import cycle, islice, product
from typing import Any, Generator, List, Optional, Tuple, Type

import numpy as np


class SpecFuzzer:
  """Fuzzer for envpool specs objects. Constructs a generator that
  yields from some test values, and the batch dim of the return value
  can be specified at each step.

  Supports the following strategies:

     - permute()
        Generate permutations of fuzzed values for each keys.

     - zip()
        Generate fuzzed instances in lock-step, i.e. running
        zip(*test_values).
        This is faster but doesn't cover all the different scenarios.

     - random()
        Generate random value within the value_range of the given dtype.

  And the following dtypes: Bool, Int, Float

  """

  # Interesting integer test values from
  # fuzzdb/attack/integer-overflow/integer-overflows.txt
  # NOTE: current API cannot handle the commented out values below
  INT_TEST_VALS = [
    # -1,
    0,
    256,
    4096,
    1073741823,
    2147483646,
    2147483647,
    2147483648,
    # 4294967294,
    # 4294967295,
    65536,
    1048576,
  ]

  # Interesting float test values from protofuzz
  FLOAT_TEST_VALS = [
    0.0,
    -1.0,
    1.0,
    -1231231231231.0123,
    123123123123123.123,
  ]

  BOOL_TEST_VALS = [True, False]

  MAX_TEST_VAL_LEN = len(INT_TEST_VALS)

  def __init__(self, spec: OrderedDict):
    self.spec = spec

  def get_batched_spec(
    self, gen: Generator[Tuple, None, None], size: int, num_env: int
  ) -> Tuple[OrderedDict, bool]:
    """Get n value from the generator, then batch it.

    Returns:
      a tuple of the batched spec with fuzz values, and a status indicating
    whether the generator is depleted.

    NOTE: assumes that all keys in the spec has a batch dim at 0.
    """
    depleted = False
    try:
      gen_value = next(gen)
    except (StopIteration, GeneratorExit):
      return OrderedDict(), True
    gen_values = [gen_value]
    for _ in range(size - 1):
      try:
        gen_value = next(gen)
      except (StopIteration, GeneratorExit):
        depleted = True
      gen_values.append(gen_value)
    gen_values = list(zip(*gen_values))  # transpose the values
    batched_values = []
    for v, (dtype, shape) in zip(gen_values, self.spec.values()):
      shape = copy(shape)
      is_shared = len(shape) == 0
      if is_shared:
        shape = [-1]
      assert shape[0] == -1, f"Non batched shape: {shape}"
      vs = np.concatenate(v, axis=0)
      shape[0] = size
      assert vs.shape == tuple(shape), f"Shape mismatch! {vs.shape} vs {shape}"
      assert vs.dtype == dtype, f"Dtype mismatch! {vs.dtype} vs {dtype}"
      if is_shared:
        vs = vs[:num_env]
      batched_values.append(vs)
    return OrderedDict(zip(self.spec.keys(), batched_values)), depleted

  def permute(self,
              limit: Optional[int] = None) -> Generator[Tuple, None, None]:
    spec_value_gen = product(
      *[
        self.gen_test_value(dtype, copy(shape))
        for dtype, shape in self.spec.values()
      ]
    )
    if limit is not None:
      yield from islice(spec_value_gen, limit)
    else:
      yield from spec_value_gen

  def zip(self, limit: Optional[int] = None) -> Generator[Tuple, None, None]:
    spec_value_gen = zip(
      *[
        cycle(self.gen_test_value(dtype, copy(shape)))
        for dtype, shape in self.spec.values()
      ]
    )
    limit = min(limit or self.MAX_TEST_VAL_LEN, self.MAX_TEST_VAL_LEN)
    yield from islice(spec_value_gen, limit)

  def random(
    self,
    limit: Optional[int] = None,
    value_range: Optional[Tuple[int, int]] = None
  ) -> Generator[Tuple, None, None]:
    spec_value_gen = zip(
      *[
        self.gen_random_value(dtype, copy(shape), value_range=value_range)
        for dtype, shape in self.spec.values()
      ]
    )
    if limit is not None:
      yield from islice(spec_value_gen, limit)
    else:
      yield from spec_value_gen

  def gen_test_value(
    self,
    dtype: Type,
    shape: List,
  ) -> Generator[np.ndarray, None, None]:
    if len(shape) == 0:
      shape = [1]
    elif shape[0] == -1:
      shape[0] = 1
    gen: Any
    if "bool" in str(dtype):
      gen = self.BOOL_TEST_VALS
    elif "int" in str(dtype):
      gen = self.INT_TEST_VALS
    elif "float" in str(dtype):
      gen = self.FLOAT_TEST_VALS
    else:
      raise NotImplementedError(f"{dtype} is not supported!")
    for v in gen:
      yield np.full(shape, v, dtype=dtype)

  def gen_random_value(
    self,
    dtype: Type,
    shape: List,
    value_range: Optional[Tuple[int, int]] = None
  ) -> Generator[np.ndarray, None, None]:
    while True:
      if len(shape) == 0:
        shape = [1]
      elif shape[0] == -1:
        shape[0] = 1
      if "bool" in str(dtype):
        yield np.random.randint(2, size=shape).astype("bool")
      elif "int" in str(dtype):
        type_info = np.iinfo(dtype)
        # NOTE: current API doesn't handles negative int
        if value_range is not None:
          low, high = value_range
        else:
          low, high = 0, type_info.max
        yield np.random.randint(low=low, high=high, dtype=dtype, size=shape)
      elif "float" in str(dtype):
        ftype_info = np.finfo(dtype)
        if value_range is not None:
          low, high = value_range
        else:
          low, high = ftype_info.min, ftype_info.max
        yield np.random.uniform(low=low, high=high, size=shape).astype(dtype)
      else:
        raise NotImplementedError(f"{dtype} is not supported!")
