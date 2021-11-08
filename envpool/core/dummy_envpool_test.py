# Copyright 2021 Garena Online Private Limited
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
"""Unit test for envpool base class."""

import numpy as np
from absl.testing import absltest

from envpool.core.dummy_envpool import _DummyEnvPool, _DummyEnvSpec


class _DummyEnvPoolTest(absltest.TestCase):

  def test_config(self) -> None:
    default_conf = _DummyEnvSpec._default_config_values
    self.assertTrue(isinstance(default_conf, tuple))
    config_keys = _DummyEnvSpec._config_keys
    self.assertTrue(isinstance(config_keys, list))
    self.assertEqual(len(default_conf), len(config_keys))
    self.assertTrue(set(["abc", "xyz", "123"]).issubset(config_keys))

  def test_spec(self) -> None:
    conf = _DummyEnvSpec._default_config_values
    env_spec = _DummyEnvSpec(conf)
    state_spec = env_spec._state_spec
    action_spec = env_spec._action_spec
    self.assertTrue(isinstance(state_spec, tuple))
    self.assertTrue(isinstance(action_spec, tuple))

    self.assertTrue(
      set(["obs:players.abc",
           "info:players.cde"]).issubset(env_spec._state_keys),
    )
    self.assertTrue(
      set(["players.do1", "players.do2"]).issubset(env_spec._action_keys),
    )

  def test_envpool(self) -> None:
    conf = _DummyEnvSpec._default_config_values
    env_spec = _DummyEnvSpec(conf)
    env = _DummyEnvPool(env_spec)
    state = env._recv()
    self.assertTrue(isinstance(state, list))
    self.assertTrue(all([isinstance(s, np.ndarray) for s in state]))
    state = dict(zip(env._state_keys, state))
    self.assertEqual(tuple(state["obs:players.abc"].shape), (2,))
    self.assertEqual(tuple(state["info:players.cde"].shape), (1, 3))
    np.testing.assert_array_equal(state["obs:players.abc"], np.array([2, 10]))
    action = [
      np.arange(4),
      np.arange(2),
      np.array(
        [
          [[10, 11, 12, 13], [0, 1, 2, 3]],
          [[110, 111, 112, 113], [3, 2, 1, 0]]
        ],
        dtype=np.int32
      ),
      np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    ]
    env._send(action)


if __name__ == "__main__":
  absltest.main()
