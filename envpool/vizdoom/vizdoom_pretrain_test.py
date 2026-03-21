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
"""Test Vizdoom env by well-trained RL agents."""

import multiprocessing as mp
import os
import queue
import shutil
from typing import Optional, Tuple

import numpy as np
import torch
from absl import logging
from absl.testing import absltest
from tianshou.data import Batch
from tianshou.policy import C51Policy

import envpool.vizdoom.registration  # noqa: F401
from envpool.atari.atari_network import C51
from envpool.registration import make_gym

# try:
#   import cv2
# except ImportError:
#   cv2 = None


def _get_map_path(path: str) -> str:
  return os.path.join("envpool", "vizdoom", "maps", path)


def _cleanup_runtime_dir() -> None:
  if os.path.isdir("_vizdoom"):
    shutil.rmtree("_vizdoom")
  elif os.path.exists("_vizdoom"):
    os.remove("_vizdoom")


def _eval_c51_impl(
  task: str,
  resume_path: str,
  num_envs: int = 10,
  seed: int = 0,
  cfg_path: Optional[str] = None,
  reward_config: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
  task_id = "".join([g.capitalize() for g in task.split("_")]) + "-v1"
  kwargs = {
    "num_envs": num_envs,
    "seed": seed,
    "wad_path": _get_map_path(task + ".wad"),
    "use_combined_action": True,
  }
  if cfg_path is None:
    kwargs.update(cfg_path=_get_map_path(task + ".cfg"))
  else:
    kwargs.update(cfg_path=cfg_path)
  if reward_config is not None:
    kwargs.update(reward_config=reward_config)
  _cleanup_runtime_dir()
  env = make_gym(task_id, **kwargs)

  state_shape = env.observation_space.shape
  action_shape = env.action_space.n
  device = "cuda" if torch.cuda.is_available() else "cpu"
  np.random.seed(seed)
  torch.manual_seed(seed)
  logging.info(state_shape)
  net = C51(*state_shape, action_shape, 51, device)  # type: ignore
  optim = torch.optim.Adam(net.parameters(), lr=1e-4)

  policy = C51Policy(
    net, optim, 0.99, 51, -10, 10, 3, target_update_freq=500
  ).to(device)
  policy.load_state_dict(torch.load(resume_path, map_location=device))
  policy.eval()
  ids = np.arange(num_envs)
  reward = np.zeros(num_envs)
  length = np.zeros(num_envs)
  obs, _ = env.reset()
  for _ in range(555):
    if np.random.rand() < 0.05:
      act = np.random.randint(action_shape, size=len(ids))
    else:
      act = policy(Batch(obs=obs, info={})).act
    obs, rew, terminated, truncated, info = env.step(act, ids)
    done = np.logical_or(terminated, truncated)
    ids = np.asarray(info["env_id"])
    reward[ids] += rew
    length[ids] += 1
    obs = obs[~done]
    ids = ids[~done]
    if len(ids) == 0:
      break

  logging.info(f"Mean reward of {task}: {reward.mean()} ± {reward.std()}")
  logging.info(f"Mean length of {task}: {length.mean()} ± {length.std()}")
  env.close()
  _cleanup_runtime_dir()
  return reward, length


def _eval_c51_subprocess(
  result_queue: mp.Queue,
  task: str,
  resume_path: str,
  cfg_path: Optional[str],
  reward_config: Optional[dict],
) -> None:
  reward, length = _eval_c51_impl(
    task,
    resume_path,
    cfg_path=cfg_path,
    reward_config=reward_config,
  )
  result_queue.put((reward, length))


class _VizdoomPretrainTest(absltest.TestCase):

  def get_path(self, path: str) -> str:
    return _get_map_path(path)

  def cleanup_runtime_dir(self) -> None:
    _cleanup_runtime_dir()

  def eval_c51(
    self,
    task: str,
    resume_path: str,
    num_envs: int = 10,
    seed: int = 0,
    cfg_path: Optional[str] = None,
    reward_config: Optional[dict] = None,
  ) -> Tuple[np.ndarray, np.ndarray]:
    return _eval_c51_impl(
      task,
      resume_path,
      num_envs=num_envs,
      seed=seed,
      cfg_path=cfg_path,
      reward_config=reward_config,
    )

  def test_d1(self) -> None:
    model_path = os.path.join("envpool", "vizdoom", "policy-d1.pth")
    self.assertTrue(os.path.exists(model_path))
    _, length = self.eval_c51("D1_basic", model_path)
    self.assertGreaterEqual(length.mean(), 500)

  def test_d3(self) -> None:
    model_path = os.path.join("envpool", "vizdoom", "policy-d3.pth")
    self.assertTrue(os.path.exists(model_path))
    # test with customized config
    with open(self.get_path("D3_battle.cfg")) as f:
      cfg = f.read()
    with open("d3.cfg", "w") as f:
      f.write(cfg.replace("hud = false", "hud = true"))
    try:
      ctx = mp.get_context("spawn")
      result_queue: mp.Queue = ctx.Queue()
      proc = ctx.Process(
        target=_eval_c51_subprocess,
        args=(
          result_queue,
          "D3_battle",
          model_path,
          "d3.cfg",
          {
            "KILLCOUNT": [1, 0]
          },
        ),
      )
      proc.start()
      try:
        # Full bazel runs may overlap this spawned evaluation with other
        # long-running pretrain tests on shared devbox CPUs.
        reward, length = result_queue.get(timeout=360)
      except queue.Empty:
        proc.terminate()
        proc.join(timeout=5)
        self.fail("Timed out waiting for D3 subprocess result")
      proc.join(timeout=30)
      if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        self.fail("D3 subprocess did not exit after producing a result")
      self.assertEqual(proc.exitcode, 0)
      result_queue.close()
      result_queue.join_thread()
    finally:
      if os.path.exists("d3.cfg"):
        os.remove("d3.cfg")
    self.assertGreaterEqual(reward.mean(), 20)


if __name__ == "__main__":
  absltest.main()
