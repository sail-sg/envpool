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

import numpy as np

import envpool


def make_env() -> None:
  # EnvPool now supports gym and dm_env API.
  # You need to manually specify which API you want to make.
  # For example, make dm_env:
  env_dm = envpool.make("Pong-v5", env_type="dm")
  # it is the same as
  env_dm0 = envpool.make_dm("Pong-v5")
  # and gym:
  env_gym = envpool.make("Pong-v5", env_type="gym")
  # it is the same as
  env_gym0 = envpool.make_gym("Pong-v5")
  # For easier debugging, you can directly print env,
  # and it includes all available configurations to this env
  print(env_dm)
  print(env_gym)
  assert str(env_dm) == str(env_dm0)
  assert str(env_gym) == str(env_gym0)
  # To use this configuration, just add these kwargs into `make` function.
  # For example, open an envpool that contains 4 envs:
  env = envpool.make_gym("Pong-v5", num_envs=4)
  print(env)


def make_spec() -> None:
  # In the past, we need to make a fake env to get the actual observation
  # and action space.
  # But in envpool, you can do this by `make_spec`.
  # It can accept the same kwargs as `make`.
  spec = envpool.make_spec("Pong-v5", num_envs=4)
  print(spec)
  # You can get both observation and action space from spec
  gym_obs_space = spec.observation_space
  gym_act_space = spec.action_space
  dm_obs_spec = spec.observation_spec()
  dm_act_spec = spec.action_spec()
  np.testing.assert_allclose(gym_obs_space.high, 255)
  assert gym_act_space.n == 6  # 6 action in Pong
  np.testing.assert_allclose(dm_obs_spec.obs.maximum, 255)
  assert dm_act_spec.num_values, 6


if __name__ == "__main__":
  make_env()
  make_spec()
