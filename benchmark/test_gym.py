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

"""Benchmark EnvPool against Gym vector environments."""

import argparse
import time

import ale_py
import gymnasium as gym
import tqdm
from atari_wrappers import wrap_deepmind


def make_vector_env(num_envs, async_, make_env):
    """Create a Gym vector environment."""
    if async_:
        vector_env_cls = gym.vector.AsyncVectorEnv
    else:
        vector_env_cls = gym.vector.SyncVectorEnv
    return vector_env_cls([make_env for _ in range(num_envs)])


def run(env, num_envs, total_step, async_):
    """Benchmark a vectorized environment."""
    if env == "atari":
        gym.register_envs(ale_py)
        task_id = "ALE/Pong-v5"
        frame_skip = 4
        make_kwargs = {"frameskip": 1}
        if num_envs == 1:
            env = wrap_deepmind(
                gym.make(task_id, **make_kwargs),
                episode_life=False,
                clip_rewards=False,
                frame_stack=4,
            )
        else:
            env = make_vector_env(
                num_envs,
                async_,
                lambda: wrap_deepmind(
                    gym.make(task_id, **make_kwargs),
                    episode_life=False,
                    clip_rewards=False,
                    frame_stack=4,
                ),
            )
    elif env == "mujoco":
        task_id = "Ant-v5"
        frame_skip = 5
        if num_envs == 1:
            env = gym.make(task_id)
        else:
            env = make_vector_env(num_envs, async_, lambda: gym.make(task_id))
    elif env == "box2d":
        task_id = "LunarLander-v3"
        frame_skip = 1
        if num_envs == 1:
            env = gym.make(task_id)
        else:
            env = make_vector_env(num_envs, async_, lambda: gym.make(task_id))
    else:
        raise NotImplementedError(f"Unknown env {env}")
    env.reset(seed=0)
    action = env.action_space.sample()
    terminated = truncated = False
    t = time.time()
    for _ in tqdm.trange(total_step):
        if num_envs == 1:
            if terminated or truncated:
                terminated = truncated = False
                env.reset()
            else:
                _, _, terminated, truncated, _ = env.step(action)
        else:
            env.step(action)
    env.close()
    print(f"FPS = {frame_skip * total_step * num_envs / (time.time() - t):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="atari", choices=["atari", "mujoco", "box2d"]
    )
    parser.add_argument("--async_", action="store_true")
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--total-step", type=int, default=5000)
    args = parser.parse_args()
    print(args)
    run(args.env, args.num_envs, args.total_step, args.async_)
