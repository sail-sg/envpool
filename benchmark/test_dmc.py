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

import numpy as np
import tqdm
from dm_control import suite

import envpool


def run_dmc(env, action, frame_skip, total_step):
  ts = env.reset()
  t = time.time()
  for i in tqdm.trange(total_step):
    if ts.discount == 0:
      ts = env.reset()
    else:
      ts = env.step(action[i])
  fps = frame_skip * total_step / (time.time() - t)
  print(f"FPS(dmc) = {fps:.2f}")
  return fps


def run_envpool(env, action, frame_skip, total_step):
  ts = env.reset()
  t = time.time()
  for i in tqdm.trange(total_step):
    if ts.discount[0] == 0:
      ts = env.reset()
    else:
      ts = env.step(action[i:i + 1])
  fps = frame_skip * total_step / (time.time() - t)
  print(f"FPS(envpool) = {fps:.2f}")
  return fps


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--domain", type=str, default="cheetah")
  parser.add_argument("--task", type=str, default="run")
  parser.add_argument("--total-step", type=int, default=200000)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()
  print(args)

  # dmc
  env = suite.load(args.domain, args.task, {"random": args.seed})
  np.random.seed(args.seed)
  minimum, maximum = env.action_spec().minimum, env.action_spec().maximum
  action = np.array(
    [
      np.random.uniform(low=minimum, high=maximum)
      for _ in range(args.total_step)
    ]
  )
  frame_skip = env._n_sub_steps

  fps_dmc = run_dmc(env, action, frame_skip, args.total_step)

  time.sleep(3)

  # envpool
  domain_name = "".join([i.capitalize() for i in args.domain.split("_")])
  task_name = "".join([i.capitalize() for i in args.task.split("_")])
  env = envpool.make_dm(f"{domain_name}{task_name}-v1", num_envs=1)
  assert env.spec.config.frame_skip == frame_skip

  fps_envpool = run_envpool(env, action, frame_skip, args.total_step)
  print(f"EnvPool Speedup: {fps_envpool / fps_dmc:.2f}x")
