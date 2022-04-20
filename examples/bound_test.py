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

import gym

# A simple test on the elementwise boundary
import envpool

if __name__ == "__main__":
  env = gym.make("CartPole-v1")
  print(env.observation_space)
  print(env.observation_space.high)
  print(env.observation_space.low)

  test_envs = envpool.make("CartPole-v1", num_envs=100, env_type="gym")
  print(test_envs.observation_space)
  print(test_envs.observation_space.high)
  print(test_envs.observation_space.low)
