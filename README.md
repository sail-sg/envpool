<div align="center">
  <a href="http://envpool.readthedocs.io"><img width="666px" height="auto" src="docs/_static/images/envpool-logo.png"></a>
</div>

---

[![PyPI](https://img.shields.io/pypi/v/envpool)](https://pypi.org/project/envpool/)
[![Read the Docs](https://img.shields.io/readthedocs/envpool)](https://envpool.readthedocs.io/)
[![Unittest](https://github.com/sail-sg/envpool/workflows/Bazel%20Build%20and%20Test/badge.svg?branch=master)](https://github.com/sail-sg/envpool/actions)
[![GitHub issues](https://img.shields.io/github/issues/sail-sg/envpool)](https://github.com/sail-sg/envpool/issues)
[![GitHub stars](https://img.shields.io/github/stars/sail-sg/envpool)](https://github.com/sail-sg/envpool/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/sail-sg/envpool)](https://github.com/sail-sg/envpool/network)
[![GitHub license](https://img.shields.io/github/license/sail-sg/envpool)](https://github.com/sail-sg/envpool/blob/master/LICENSE)

**EnvPool** is a highly parallel reinforcement learning environment execution engine which significantly outperforms existing environment executors. With a curated design dedicated to the RL use case, we leverage techniques of a general asynchronous execution model, implemented with C++ thread pool on the environment execution.

Here are EnvPool's several highlights:

- Compatible with OpenAI `gym` APIs and DeepMind `dm_env` APIs;
- Manage a pool of envs, interact with the envs in batched APIs by default;
- Synchronous execution API and asynchronous execution API;
- Easy C++ developer API to add new envs;
- **1 Million** Atari frames per second simulation with 256 CPU cores, **~13x** throughput of Python subprocess-based vector env;
- **~3x** throughput of Python subprocess-based vector env on low resource setup like 12 CPU cores;
- Comparing with existing GPU-based solution ([Brax](https://github.com/google/brax) / [Isaac-gym](https://developer.nvidia.com/isaac-gym)), EnvPool is a **general** solution for various kinds of speeding-up RL environment parallelization;
- Compatible with some existing RL libraries, e.g., [Tianshou](https://github.com/thu-ml/tianshou).

## Installation

### PyPI

EnvPool is currently hosted on [PyPI](https://pypi.org/project/envpool/). It requires Python >= 3.6.

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

### Build From Source

Please refer to the [guideline](https://envpool.readthedocs.io/en/latest/pages/build.html).

## Documentation

The tutorials and API documentation are hosted on [envpool.readthedocs.io](https://envpool.readthedocs.io).

The example scripts are under [examples/](https://github.com/sail-sg/envpool/tree/master/examples) folder.

## Supported Environments

We're in the progress of open-sourcing all available envs from our internal version, stay tuned.

- [x] Atari via ALE
- [ ] Single/Multi players Vizdoom
- [ ] Classic RL envs, including CartPole, MountainCar, ... 

## Benchmark Results
We perform our benchmarks with ALE Atari environment (with environment wrappers) on different hardware setups, including a TPUv3-8 virtual machine (VM) of 96 CPU cores and 2 NUMA nodes, and an NVIDIA DGX-A100 of 256 CPU cores with 8 NUMA nodes. Baselines include 1) naive Python for-loop; 2) the most popular RL environment parallelization execution by Python subprocess, e.g., [gym.vector_env](https://github.com/openai/gym/blob/master/gym/vector/vector_env.py); 3) to our knowledge, the fastest RL environment executor [Sample Factory](https://github.com/alex-petrenko/sample-factory) before EnvPool. 

We report EnvPool performance with sync mode, async mode and NUMA + async mode, compared with the baselines on different number of workers (i.e., number of CPU cores). As we can see from the results, EnvPool achieves significant improvements over the baselines on all settings. On the high-end setup, EnvPool achieves 1 Million frames per second on 256 CPU cores, which is 13.3x of the `gym.vector_env` baseline. On a typical PC setup with 12 CPU cores, EnvPool's throughput is 2.8x of `gym.vector_env`.

Our benchmark script is in [examples/benchmark.py](https://github.com/sail-sg/envpool/blob/master/examples/benchmark.py). We choose 4 types of system:

- Personal laptop: 12 core `Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz` 
- TPU-VM: 96 core `Intel(R) Xeon(R) CPU @ 2.00GHz`
- Apollo: 96 core `AMD EPYC 7352 24-Core Processor`
- DGX-A100: 256 core `AMD EPYC 7742 64-Core Processor`

| Highest FPS          | i7-8750H laptop (12) | TPU-VM (96) | Apollo (96) | DGX-A100 (256) |
| -------------------- | -------------------- | ----------- | ----------- | -------------- |
| For-loop             | 4876                 | 3817        | 4053        | 4336           |
| Subprocess           | 18249                | 42885       | 19560       | 79509          |
| Sample Factory       | 27035                | 192074      | 262963      | 639389         |
| EnvPool (sync)       | 40791                | 175938      | 159191      | 470170         |
| EnvPool (async)      | **50513**            | 352243      | 410941      | 845537         |
| EnvPool (numa+async) | /                    | **367799**  | **458414**  | **1060371**    |

![](https://i.imgur.com/wHu7m4C.png)

![](https://i.imgur.com/JP5RApq.png)

## API Usage

The following content shows both synchronous and asynchronous API usage of EnvPool. You can also run the full script at [examples/env_step.py](https://github.com/sail-sg/envpool/blob/master/examples/env_step.py)

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

Under the synchronous mode, `envpool` closely resembles `openai-gym`/`dm-env`. It has the `reset` and `step` function with the same meaning. There is one exception though, in `envpool` batch interaction is the default. Therefore, during creation of the envpool, there is a `num_envs` argument that denotes how many envs you like to run in parallel.

```python
env = envpool.make("Pong-v5", env_type="gym", num_envs=100)
```

The first dimension of `action` passed to the step function should be equal to `num_envs`.

```python
act = np.zeros(100, dtype=int)
```

You don't need to manually reset one environment when any of `done` is true, instead, all envs in `envpool` has enabled auto-reset by default.

### Asynchronous API

```python
import envpool
import numpy as np

# make asynchronous 
env = envpool.make("Pong-v5", env_type="gym", num_envs=64, batch_size=16)
env.async_reset()  # send the initial reset signal to all envs
while True:
    obs, rew, done, info = env.recv()
    action = np.random.randint(batch_size, size=len(info.env_id))
    env.send(action, env_id)
```

In the asynchronous mode, the `step` function is splitted into two part, namely the `send`/`recv` functions. `send` takes two arguments, a batch of action, and the corresponding `env_id` that each action should be sent to. Unlike `step`, `send` does not wait for the envs to execute and return the next state, it returns immediately after the actions are fed to the envs. (The reason why it is called async mode).

```python
env.send(action, env_id)
```
To get the "next states", we need to call the `recv` function. However, `recv` does not guarantee that you will get back the "next states" of the envs that you just called `send` on. Instead, whatever envs finishes execution first gets `recv`ed first.

```python
state = env.recv()
```

Besides `num_envs`, there's one more argument `batch_size`. While `num_envs` defines how many envs in total is being managed by the `envpool`, `batch_size` defines the number of envs involved each time we interact with `envpool`. e.g. There're 64 envs executing in the `envpool`, `send` and `recv` each time interacts with a batch of 16 envs.

```python
envpool.make("Pong-v5", env_type="gym", num_envs=64, batch_size=16)
```

## Contributing

EnvPool is still under development. More environments are going to be added and we always welcome contributions to help EnvPool better. If you would like to contribute, please check out our [contribution guideline](https://envpool.readthedocs.io/en/latest/pages/contributing.html).

## License

EnvPool is under Apache2 license.

Other third party source-code and data are under their corresponding licenses.

We do not include their source-code and data in this repo.

## Citing EnvPool

If you find EnvPool useful, please cite it in your publications.

[Coming soon!]


## Disclaimer
This is not an official Sea Limited or Garena Online Private Limited product.

