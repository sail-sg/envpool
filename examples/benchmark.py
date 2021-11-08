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

import argparse
import time

import numpy as np

import envpool

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", type=str, default="Pong-v5")
  parser.add_argument("--num-envs", type=int, default=645)
  parser.add_argument("--batch-size", type=int, default=248)
  # num_threads == 0 means to let envpool itself determine
  parser.add_argument("--num-threads", type=int, default=0)
  # thread_affinity_offset == -1 means no thread affinity
  parser.add_argument("--thread-affinity-offset", type=int, default=0)
  parser.add_argument("--total-iter", type=int, default=50000)
  args = parser.parse_args()
  env = envpool.make_gym(
    "Pong-v5",
    num_envs=args.num_envs,
    batch_size=args.batch_size,
    num_threads=args.num_threads,
    thread_affinity_offset=args.thread_affinity_offset,
  )
  env.async_reset()
  action = np.ones(args.batch_size, dtype=np.int32)
  t = time.time()
  for _ in range(args.total_iter):
    info = env.recv()[-1]
    env.send(action, info["env_id"])
  duration = time.time() - t
  fps = args.total_iter * args.batch_size / duration * 4
  print(f"Duration = {duration:.2f}s")
  print(f"EnvPool FPS = {fps:.2f}")
