from typing import Any, List, Tuple

import numpy as np
from absl import logging
from absl.testing import absltest

from envpool.core.dummy_envpool import _DummyEnvPool, _DummyEnvSpec


class _DummyEnvPoolTest(absltest.TestCase):

  def test_config(self) -> None:
    default_conf = _DummyEnvSpec._default_config_values
    self.assertTrue(isinstance(default_conf, tuple))
    config_keys = _DummyEnvSpec._config_keys
    self.assertTrue(isinstance(config_keys, list))
    self.assertEqual(len(default_conf), len(config_keys))
    logging.info(list(zip(config_keys, default_conf)))

  def test_spec(self) -> None:
    conf = _DummyEnvSpec._default_config_values
    env_spec = _DummyEnvSpec(conf)
    state_spec = env_spec._state_spec
    action_spec = env_spec._action_spec
    self.assertTrue(isinstance(state_spec, list))
    self.assertTrue(isinstance(action_spec, list))

    def all_string(spec: List[Tuple[str, Any]]) -> bool:
      return all(map(lambda x: isinstance(x[0], str), spec))

    self.assertTrue(all_string(state_spec))
    self.assertTrue(all_string(action_spec))
    logging.info("dummy envpool state spec: %s", state_spec)
    logging.info("dummy envpool action spec: %s", action_spec)

  def test_envpool(self) -> None:
    conf = _DummyEnvSpec._default_config_values
    env_spec = _DummyEnvSpec(conf)
    env = _DummyEnvPool(env_spec)
    state = env._recv()
    self.assertTrue(isinstance(state, list))
    self.assertTrue(all([isinstance(s, np.ndarray) for s in state]))
    self.assertEqual(tuple(state[0].shape), (2,))
    self.assertEqual(tuple(state[1].shape), (1, 3))
    np.testing.assert_array_equal(state[0], np.array([2, 10]))
    action = [
      np.array([[[10, 11, 12, 13], [0, 1, 2, 3]]], dtype=np.int32),
      np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    ]
    env._send(action)
    logging.info("dummy envpool state: %s", state)


if __name__ == "__main__":
  absltest.main()
