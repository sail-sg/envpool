# Copyright 2022 Garena Online Private Limited
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

import argparse
import time

import gym
import tqdm
from atari_wrappers import wrap_deepmind


def run(env, numenv, total_step, async_):
  if env == "atari":
    task_id = "PongNoFrameskip-v4"
    frame_skip = 4
    env = gym.vector.make(
      task_id, numenv, async_, lambda e:
      wrap_deepmind(e, episode_life=False, clip_rewards=False, frame_stack=4)
    )
  elif env == "mujoco":
    task_id = "Ant-v3"
    frame_skip = 5
    env = gym.vector.make(task_id, numenv, async_)
  else:
    raise NotImplementedError(f"Unknown env {env}")
  env.seed(0)
  env.reset()
  action = env.action_space.sample()
  t = time.time()
  for _ in tqdm.trange(total_step):
    env.step(action)
  print(f"FPS = {frame_skip * total_step * numenv / (time.time() - t):.2f}")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--env", type=str, default="atari", choices=["atari", "mujoco"]
  )
  parser.add_argument("--async_", action="store_true")
  parser.add_argument("--numenv", type=int, default=10)
  parser.add_argument("--total-step", type=int, default=5000)
  args = parser.parse_args()
  print(args)
  run(args.env, args.numenv, args.total_step, args.async_)
