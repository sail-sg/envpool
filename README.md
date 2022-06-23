<div align="center">
  <a href="http://envpool.readthedocs.io"><img width="666px" height="auto" src="https://envpool.readthedocs.io/en/latest/_static/envpool-logo.png"></a>
</div>


---

[![PyPI](https://img.shields.io/pypi/v/envpool)](https://pypi.org/project/envpool/) [![Downloads](https://static.pepy.tech/personalized-badge/envpool?period=total&units=international_system&left_color=grey&right_color=orange&left_text=PyPI%20Download)](https://pepy.tech/project/envpool) [![Read the Docs](https://img.shields.io/readthedocs/envpool)](https://envpool.readthedocs.io/) [![Unittest](https://github.com/sail-sg/envpool/workflows/Bazel%20Build%20and%20Test/badge.svg?branch=main)](https://github.com/sail-sg/envpool/actions) [![GitHub issues](https://img.shields.io/github/issues/sail-sg/envpool)](https://github.com/sail-sg/envpool/issues) [![GitHub stars](https://img.shields.io/github/stars/sail-sg/envpool)](https://github.com/sail-sg/envpool/stargazers) [![GitHub forks](https://img.shields.io/github/forks/sail-sg/envpool)](https://github.com/sail-sg/envpool/network) [![GitHub license](https://img.shields.io/github/license/sail-sg/envpool)](https://github.com/sail-sg/envpool/blob/main/LICENSE)

**EnvPool** is a C++-based batched environment pool with pybind11 and thread pool. It has high performance (\~1M raw FPS with Atari games, \~3M raw FPS with Mujoco simulator on DGX-A100) and compatible APIs (supports both gym and dm\_env, both sync and async, both single and multi player environment). Currently it supports:

- [x] [Atari games](https://envpool.readthedocs.io/en/latest/env/atari.html)
- [x] [Mujoco (gym)](https://envpool.readthedocs.io/en/latest/env/mujoco_gym.html)
- [x] [Classic control RL envs](https://envpool.readthedocs.io/en/latest/env/classic_control.html): CartPole, MountainCar, Pendulum, Acrobot
- [x] [Toy text RL envs](https://envpool.readthedocs.io/en/latest/env/toy_text.html): Catch, FrozenLake, Taxi, NChain, CliffWalking, Blackjack
- [x] [ViZDoom single player](https://envpool.readthedocs.io/en/latest/env/vizdoom.html)
- [x] [DeepMind Control Suite](https://envpool.readthedocs.io/en/latest/env/dm_control.html)
- [ ] [Box2D](https://envpool.readthedocs.io/en/latest/env/box2d.html)
- [ ] Procgen
- [ ] Minigrid

Here are EnvPool's several highlights:

- Compatible with OpenAI `gym` APIs and DeepMind `dm_env` APIs;
- Manage a pool of envs, interact with the envs in batched APIs by default;
- Support both synchronous execution and asynchronous execution;
- Support both single player and multi-player environment;
- Easy C++ developer API to add new envs: [Customized C++ environment integration](https://envpool.readthedocs.io/en/latest/content/new_env.html);
- Free **\~2x** speedup with only single environment;
- **1 Million** Atari frames / **3 Million** Mujoco steps per second simulation with 256 CPU cores, **~20x** throughput of Python subprocess-based vector env;
- **~3x** throughput of Python subprocess-based vector env on low resource setup like 12 CPU cores;
- Comparing with existing GPU-based solution ([Brax](https://github.com/google/brax) / [Isaac-gym](https://developer.nvidia.com/isaac-gym)), EnvPool is a **general** solution for various kinds of speeding-up RL environment parallelization;
- XLA support with Jax jit function: [XLA Interface](https://envpool.readthedocs.io/en/latest/content/xla_interface.html);
- Compatible with some existing RL libraries, e.g., [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [Tianshou](https://github.com/thu-ml/tianshou), [ACME](https://github.com/deepmind/acme), [CleanRL](https://github.com/vwxyzjn/cleanrl), or [rl\_games](https://github.com/Denys88/rl_games).
  - Stable-Baselines3 [`Pendulum-v1` example](https://github.com/sail-sg/envpool/blob/main/examples/sb3_examples/ppo.py);
  - Tianshou [`CartPole` example](https://github.com/sail-sg/envpool/blob/main/examples/tianshou_examples/cartpole_ppo.py), [`Pendulum-v1` example](https://github.com/sail-sg/envpool/blob/main/examples/tianshou_examples/pendulum_ppo.py), [Atari example](https://github.com/thu-ml/tianshou/tree/master/examples/atari#envpool), [Mujoco example](https://github.com/thu-ml/tianshou/tree/master/examples/mujoco#envpool), and [integration guideline](https://tianshou.readthedocs.io/en/master/tutorials/cheatsheet.html#envpool-integration);
  - ACME [`HalfCheetah` example](https://github.com/sail-sg/envpool/blob/main/examples/acme_examples/ppo_continuous.py);
  - CleanRL [`Pong-v5` example](https://github.com/sail-sg/envpool/blob/main/examples/cleanrl_examples/ppo_atari_envpool.py) ([Solving Pong in 5 mins](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#solving-pong-in-5-minutes-with-ppo--envpool) ([tracked experiment](https://wandb.ai/costa-huang/cleanRL/runs/opk2dmta)));
  - rl\_games [Atari example](https://github.com/Denys88/rl_games/blob/master/docs/ATARI_ENVPOOL.md) (2 mins Pong and 15 mins Breakout) and [Mujoco example](https://github.com/Denys88/rl_games/blob/master/docs/MUJOCO_ENVPOOL.md) (5 mins Ant and HalfCheetah).

## Installation

### PyPI

EnvPool is currently hosted on [PyPI](https://pypi.org/project/envpool/). It requires Python >= 3.7.

You can simply install EnvPool with the following command:

```bash
$ pip install envpool
```

After installation, open a Python console and type

```python
import envpool
print(envpool.__version__)
```

If no error occurs, you have successfully installed EnvPool.

### From Source

Please refer to the [guideline](https://envpool.readthedocs.io/en/latest/content/build.html).

## Documentation

The tutorials and API documentation are hosted on [envpool.readthedocs.io](https://envpool.readthedocs.io).

The example scripts are under [examples/](https://github.com/sail-sg/envpool/tree/main/examples) folder; benchmark scripts are under [benchmark/](https://github.com/sail-sg/envpool/tree/main/benchmark) folder.

## Benchmark Results

We perform our benchmarks with ALE Atari environment `PongNoFrameskip-v4` (with environment wrappers from [OpenAI Baselines](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py)) and Mujoco environment `Ant-v3` on different hardware setups, including a TPUv3-8 virtual machine (VM) of 96 CPU cores and 2 NUMA nodes, and an NVIDIA DGX-A100 of 256 CPU cores with 8 NUMA nodes. Baselines include 1) naive Python for-loop; 2) the most popular RL environment parallelization execution by Python subprocess, e.g., [gym.vector_env](https://github.com/openai/gym/blob/master/gym/vector/vector_env.py); 3) to our knowledge, the fastest RL environment executor [Sample Factory](https://github.com/alex-petrenko/sample-factory) before EnvPool. 

We report EnvPool performance with sync mode, async mode, and NUMA + async mode, compared with the baselines on different number of workers (i.e., number of CPU cores). As we can see from the results, EnvPool achieves significant improvements over the baselines on all settings. On the high-end setup, EnvPool achieves 1 Million frames per second with Atari and 3 Million frames per second with Mujoco on 256 CPU cores, which is 14.9x / 19.6x of the `gym.vector_env` baseline. On a typical PC setup with 12 CPU cores, EnvPool's throughput is 3.1x / 2.9x of `gym.vector_env`.

|  Atari Highest FPS   | Laptop (12) | Workstation (32) | TPU-VM (96) | DGX-A100 (256) |
| :------------------: | :---------: | :--------------: | :---------: | :------------: |
|       For-loop       |    4,893    |      7,914       |    3,993    |     4,640      |
|      Subprocess      |   15,863    |      47,699      |   46,910    |     71,943     |
|    Sample-Factory    |   28,216    |     138,847      |   222,327   |    707,494     |
|    EnvPool (sync)    |   37,396    |     133,824      |   170,380   |    427,851     |
|   EnvPool (async)    | **49,439**  |   **200,428**    |   359,559   |    891,286     |
| EnvPool (numa+async) |      /      |        /         | **373,169** | **1,069,922**  |

|  Mujoco Highest FPS  | Laptop (12) | Workstation (32) | TPU-VM (96) | DGX-A100 (256) |
| :------------------: | :---------: | :--------------: | :---------: | :------------: |
|       For-loop       |   12,861    |      20,298      |   10,474    |     11,569     |
|      Subprocess      |   36,586    |     105,432      |   87,403    |    163,656     |
|    Sample-Factory    |   62,510    |     309,264      |   461,515   |   1,573,262    |
|    EnvPool (sync)    |   66,622    |     380,950      |   296,681   |    949,787     |
|   EnvPool (async)    | **105,126** |   **582,446**    |   887,540   |   2,363,864    |
| EnvPool (numa+async) |      /      |        /         | **896,830** | **3,134,287**  |

![](https://envpool.readthedocs.io/en/latest/_images/throughput.png)


Please refer to the [benchmark](https://envpool.readthedocs.io/en/latest/content/benchmark.html) page for more details.

## API Usage

The following content shows both synchronous and asynchronous API usage of EnvPool. You can also run the full script at [examples/env_step.py](https://github.com/sail-sg/envpool/blob/main/examples/env_step.py)

### Synchronous API

```python
import envpool
import numpy as np

# make gym env
env = envpool.make("Pong-v5", env_type="gym", num_envs=100)
# or use envpool.make_gym(...)
obs = env.reset()  # should be (100, 4, 84, 84)
act = np.zeros(100, dtype=int)
obs, rew, done, info = env.step(act)
```

Under the synchronous mode, `envpool` closely resembles `openai-gym`/`dm-env`. It has the `reset` and `step` functions with the same meaning. However, there is one exception in `envpool`: batch interaction is the default. Therefore, during the creation of the envpool, there is a `num_envs` argument that denotes how many envs you like to run in parallel.

```python
env = envpool.make("Pong-v5", env_type="gym", num_envs=100)
```

The first dimension of `action` passed to the step function should equal `num_envs`.

```python
act = np.zeros(100, dtype=int)
```

You don't need to manually reset one environment when any of `done` is true; instead, all envs in `envpool` have enabled auto-reset by default.

### Asynchronous API

```python
import envpool
import numpy as np

# make asynchronous
num_envs = 64
batch_size = 16
env = envpool.make("Pong-v5", env_type="gym", num_envs=num_envs, batch_size=batch_size)
action_num = env.action_space.n
env.async_reset()  # send the initial reset signal to all envs
while True:
    obs, rew, done, info = env.recv()
    env_id = info["env_id"]
    action = np.random.randint(action_num, size=batch_size)
    env.send(action, env_id)
```

In the asynchronous mode, the `step` function is split into two parts: the `send`/`recv` functions. `send` takes two arguments, a batch of action, and the corresponding `env_id` that each action should be sent to. Unlike `step`, `send` does not wait for the envs to execute and return the next state, it returns immediately after the actions are fed to the envs. (The reason why it is called async mode).

```python
env.send(action, env_id)
```
To get the "next states", we need to call the `recv` function. However, `recv` does not guarantee that you will get back the "next states" of the envs you just called `send` on. Instead, whatever envs finishes execution gets `recv`ed first.

```python
state = env.recv()
```

Besides `num_envs`, there is one more argument `batch_size`. While `num_envs` defines how many envs in total are managed by the `envpool`, `batch_size` specifies the number of envs involved each time we interact with `envpool`. e.g. There are 64 envs executing in the `envpool`, `send` and `recv` each time interacts with a batch of 16 envs.

```python
envpool.make("Pong-v5", env_type="gym", num_envs=64, batch_size=16)
```

There are other configurable arguments with `envpool.make`; please check out [EnvPool Python interface introduction](https://envpool.readthedocs.io/en/latest/content/python_interface.html).

## Contributing

EnvPool is still under development. More environments will be added, and we always welcome contributions to help EnvPool better. If you would like to contribute, please check out our [contribution guideline](https://envpool.readthedocs.io/en/latest/content/contributing.html).

## License

EnvPool is under Apache2 license.

Other third-party source-code and data are under their corresponding licenses.

We do not include their source code and data in this repo.

## Citing EnvPool

If you find EnvPool useful, please cite it in your publications.

```latex
@article{envpool,
  title={EnvPool: A Highly Parallel Reinforcement Learning Environment Execution Engine},
  author={Weng, Jiayi and Lin, Min and Huang, Shengyi and Liu, Bo and Makoviichuk, Denys and Makoviychuk, Viktor and Liu, Zichen and Song, Yufan and Luo, Ting and Jiang, Yukun and Xu, Zhongwen and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2206.10558},
  year={2022}
}
```

## Disclaimer

This is not an official Sea Limited or Garena Online Private Limited product.
