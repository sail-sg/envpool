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

from timeit import timeit
from typing import Tuple

import numpy as np
from numba import njit


@njit
def compute_gae(
  gamma: float,
  gae_lambda: float,
  value: np.ndarray,
  reward: np.ndarray,
  done: np.ndarray,
  env_id: np.ndarray,
  numenv: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  # shape of array: [T, B]
  # return returns, advantange, mask
  T, B = value.shape
  mask = (1.0 - done) * (gamma * gae_lambda)
  index_tp1 = np.zeros(numenv, np.int32) - 1
  value_tp1 = np.zeros(numenv)
  gae_tp1 = np.zeros(numenv)
  delta = reward - value
  adv = np.zeros((T, B))
  for t in range(T - 1, -1, -1):
    eid = env_id[t]
    adv[t] = delta[t] + gamma * value_tp1[eid] * (1 - done[t]) \
      + mask[t] * gae_tp1[eid]
    mask[t] = (done[t] | (index_tp1[eid] != -1))
    gae_tp1[eid] = adv[t]
    value_tp1[eid] = value[t]
    index_tp1[eid] = t
  return adv + value, adv, mask


def test_episodic_returns():
  # basic test for 1d array
  value = np.zeros([8, 1])
  done = np.array([1, 0, 0, 1, 0, 1, 0, 1.]).reshape(8, 1).astype(bool)
  rew = np.array([0, 1, 2, 3, 4, 5, 6, 7.]).reshape(8, 1)
  env_id = np.zeros([8, 1], int)
  returns, adv, mask = compute_gae(
    gamma=0.1,
    gae_lambda=1,
    value=value,
    reward=rew,
    done=done,
    env_id=env_id,
    numenv=1
  )
  ans = np.array([0, 1.23, 2.3, 3, 4.5, 5, 6.7, 7]).reshape([8, 1])
  assert np.allclose(returns, ans) and np.allclose(adv, ans)
  ref_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1]).reshape(8, 1)
  assert np.allclose(ref_mask, mask)

  # same as above, only shuffle index
  env_id = np.array([[1, 2, 0, 1], [3, 3, 1, 2]]).transpose()
  value = np.zeros([4, 2])
  done = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], bool).transpose().astype(bool)
  rew = np.array([[1, 4, 0, 3], [6, 7, 2, 5]]).transpose()
  returns, adv, mask = compute_gae(
    gamma=0.1,
    gae_lambda=1,
    value=value,
    reward=rew,
    done=done,
    env_id=env_id,
    numenv=4
  )
  ans = np.array([[1.23, 4.5, 0, 3], [6.7, 7, 2.3, 5]]).transpose()
  assert np.allclose(returns, ans) and np.allclose(adv, ans), returns
  ref_mask = np.ones([4, 2])
  assert np.allclose(ref_mask, mask)

  # check if mask correct in done=False at the end of trajectory
  env_id = np.zeros([7, 1])
  done = np.array([0, 1, 0, 1, 0, 1, 0]).reshape(7, 1).astype(bool)
  rew = np.array([7, 6, 1, 2, 3, 4, 5.]).reshape(7, 1)
  env_id = np.zeros([7, 1], int)
  value = np.zeros([7, 1])
  returns, adv, mask = compute_gae(
    gamma=0.1,
    gae_lambda=1,
    value=value,
    reward=rew,
    done=done,
    env_id=env_id,
    numenv=1
  )
  ans = np.array([7.6, 6, 1.2, 2, 3.4, 4, 5]).reshape(7, 1)
  assert np.allclose(returns, ans) and np.allclose(adv, ans)
  ref_mask = np.ones([7, 1])
  ref_mask[-1] = 0
  assert np.allclose(ref_mask, mask), mask

  done = np.array([0, 1, 0, 1, 0, 0, 1], bool).reshape(7, 1).astype(bool)
  rew = np.array([7, 6, 1, 2, 3, 4, 5.]).reshape(7, 1)
  returns, adv, mask = compute_gae(
    gamma=0.1,
    gae_lambda=1,
    value=value,
    reward=rew,
    done=done,
    env_id=env_id,
    numenv=1
  )
  ans = np.array([7.6, 6, 1.2, 2, 3.45, 4.5, 5]).reshape(7, 1)
  assert np.allclose(returns, ans) and np.allclose(adv, ans)
  ref_mask = np.ones([7, 1])
  assert np.allclose(ref_mask, mask)

  done = np.array([0, 0, 0, 1., 0, 0, 0, 1, 0, 0, 0,
                   1]).reshape([12, 1]).astype(bool)
  rew = np.array([101, 102, 103., 200, 104, 105, 106, 201, 107, 108, 109, 202])
  rew = rew.reshape([12, 1])
  value = np.array([1000, 2., 3., 4, -1, 5., 6., 7, -2, 8., 9.,
                    10]).reshape([12, 1])
  env_id = np.zeros([12, 1], int)
  returns, adv, mask = compute_gae(
    gamma=0.99,
    gae_lambda=0.95,
    value=value,
    reward=rew,
    done=done,
    env_id=env_id,
    numenv=1,
  )
  ans = np.array(
    [
      454.8344,
      376.1143,
      291.298,
      200.,
      464.5610,
      383.1085,
      295.387,
      201.,
      474.2876,
      390.1027,
      299.476,
      202.,
    ]
  ).reshape([12, 1])
  assert np.allclose(returns, ans), (returns, adv)
  ref_mask = np.ones([12, 1])
  assert np.allclose(ref_mask, mask)

  done = np.zeros([4, 3], bool)
  done[-1] = 1
  env_id = np.array([[0, 1, 2, 1], [1, 0, 1, 2], [2, 2, 0, 0]]).transpose()
  value = np.array([[-1000, 5, 9, 7], [-1000, 2, 6, 10], [-1000, 8., 3,
                                                          4]]).transpose()
  rew = np.array(
    [[101, 105, 109, 201], [104, 102, 106, 202], [107, 108, 103, 200.]]
  ).transpose()
  returns, adv, mask = compute_gae(
    gamma=0.99,
    gae_lambda=0.95,
    value=value,
    reward=rew,
    done=done,
    env_id=env_id,
    numenv=3,
  )
  ans = np.array(
    [
      [454.8344, 383.1085, 299.476, 201.],
      [464.5610, 376.1143, 295.387, 202.],
      [474.2876, 390.1027, 291.298, 200.],
    ]
  ).transpose()
  assert np.allclose(returns, ans), returns
  assert np.allclose(mask, 1)


def test_time():
  T, B, N = 128, 8, 8 * 4
  cnt = 10000
  value = np.random.rand(T, B)
  rew = np.random.rand(T, B)
  done = np.random.randint(2, size=[T, B]).astype(bool)
  env_id = np.random.randint(N, size=[T, B])

  def wrapper():
    return compute_gae(
      gamma=0.99,
      gae_lambda=0.95,
      value=value,
      reward=rew,
      done=done,
      env_id=env_id,
      numenv=N,
    )

  wrapper()  # for compile

  print(timeit(wrapper, setup=wrapper, number=cnt) / cnt)


if __name__ == "__main__":
  # tests are from tianshou unit test
  # test_episodic_returns()
  test_time()
