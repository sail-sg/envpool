# Copyright 2023 Garena Online Private Limited
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
"""Unit tests for minigrid environments check."""

from typing import Any

import numpy as np
from absl.testing import absltest

import envpool.minigrid.registration  # noqa: F401
from envpool.registration import make_gym


class _MiniGridEnvPoolTest(absltest.TestCase):

  def test_deterministic_check(
    self,
    task_id: str = "MiniGrid-Empty-5x5-v0",
    num_envs: int = 1,
    **kwargs: Any,
  ) -> None:
    env = make_gym(task_id, num_envs=num_envs, seed=0, **kwargs)
    obs, info = env.reset()
    print(obs)


if __name__ == "__main__":
  absltest.main()
