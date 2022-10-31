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
"""Test EnvPool by well-trained RL agents."""

import os

import numpy as np
import torch
from absl import logging
from absl.testing import absltest
from tianshou.data import Batch
from tianshou.policy import QRDQNPolicy

import envpool.atari.registration  # noqa: F401
from envpool.atari.atari_network import QRDQN
from envpool.registration import make_gym

# try:
#   import cv2
# except ImportError:
#   cv2 = None


class _AtariPretrainTest(absltest.TestCase):

  def eval_qrdqn(
    self,
    task: str,
    resume_path: str,
    num_envs: int = 10,
    seed: int = 0,
    target_reward: float = 0.0,
  ) -> None:
    env = make_gym(task.capitalize() + "-v5", num_envs=num_envs, seed=seed)
    state_shape = env.observation_space.shape
    action_shape = env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(seed)
    torch.manual_seed(seed)
    logging.info(state_shape)
    net = QRDQN(*state_shape, action_shape, 200, device)  # type: ignore
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    policy = QRDQNPolicy(
      net, optim, 0.99, 200, 3, target_update_freq=500
    ).to(device)
    policy.load_state_dict(torch.load(resume_path, map_location=device))
    policy.eval()
    ids = np.arange(num_envs)
    reward = np.zeros(num_envs)
    obs, _ = env.reset()
    for _ in range(25000):
      if np.random.rand() < 5e-3:
        act = np.random.randint(action_shape, size=len(ids))
      else:
        act = policy(Batch(obs=obs, info={})).act
      obs, rew, terminated, truncated, info = env.step(act, ids)
      done = np.logical_or(terminated, truncated)
      ids = np.asarray(info["env_id"])
      reward[ids] += rew
      obs = obs[~done]
      ids = ids[~done]
      if len(ids) == 0:
        break
      # if cv2 is not None:
      #   obs_all = np.zeros((84, 84 * num_envs, 3), np.uint8)
      #   for i, j in enumerate(ids):
      #     obs_all[:, 84 * j:84 * (j + 1)] = obs[i, 1:].transpose(1, 2, 0)
      #   cv2.imwrite(f"/tmp/{task}-{t}.png", obs_all)

    rew = reward.mean()
    logging.info(f"Mean reward of {task}: {rew}")
    self.assertAlmostEqual(rew, target_reward)

  def test_pong(self) -> None:
    model_path = os.path.join("envpool", "atari", "policy-pong.pth")
    self.assertTrue(os.path.exists(model_path))
    self.eval_qrdqn("pong", model_path, target_reward=20.6)

  def test_breakout(self) -> None:
    model_path = os.path.join("envpool", "atari", "policy-breakout.pth")
    self.assertTrue(os.path.exists(model_path))
    self.eval_qrdqn("breakout", model_path, target_reward=367.8)


if __name__ == "__main__":
  absltest.main()
