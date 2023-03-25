.. EnvPool documentation master file, created by
   sphinx-quickstart on Mon Oct 25 21:01:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EnvPool!
===================

**EnvPool** is a C++-based batched environment pool with pybind11 and thread
pool. It has high performance (~1M raw FPS on Atari games / ~3M FPS with Mujoco
physics engine in DGX-A100) and compatible APIs (supports both gym and dm_env,
both sync and async, both single and multi player environment).

Here are EnvPool's several highlights:

- Compatible with OpenAI ``gym`` APIs and DeepMind ``dm_env`` APIs;
- Manage a pool of envs, interact with the envs in batched APIs by default;
- Support both synchronous execution and asynchronous execution;
- Support both single player and multi-player environment;
- Easy C++ developer API to add new envs: `Customized C++ environment integration <https://envpool.readthedocs.io/en/latest/content/new_env.html>`_;
- Free **~2x** speedup with only single environment;
- **1 Million** Atari frames / **3 Million** Mujoco steps per second
  simulation with 256 CPU cores, **~20x** throughput of Python
  subprocess-based vector env;
- **~3x** throughput of Python subprocess-based vector env on low resource
  setup like 12 CPU cores;
- XLA support with JAX jit function;
- Comparing with the existing GPU-based solution
  (`Brax <https://github.com/google/brax>`_ /
  `Isaac-gym <https://developer.nvidia.com/isaac-gym>`_), EnvPool is a
  **general** solution for various kinds of speeding-up RL environment
  parallelization;
- Compatible with some existing RL libraries, e.g.,
  `Stable-Baselines3 <https://github.com/DLR-RM/stable-baselines3>`_,
  `Tianshou <https://github.com/thu-ml/tianshou>`_,
  `ACME <https://github.com/deepmind/acme>`_,
  `CleanRL <https://github.com/vwxyzjn/cleanrl>`_
  (`Solving Pong in 5 mins <https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#solving-pong-in-5-minutes-with-ppo--envpool>`_),
  `rl_games <https://github.com/Denys88/rl_games>`_ (2 mins
  `Pong <https://colab.research.google.com/drive/1iWFv0g67mWqJONoFKNWUmu3hdxn_qUf8?usp=sharing>`_, 15 mins
  `Breakout <https://colab.research.google.com/drive/1U_NxL6gSs0yRVhfl0cKl9ttRmcmMCiBS?usp=sharing>`_, 5 mins
  `Ant <https://colab.research.google.com/drive/1C9yULxU_ahQ_i6NUHCvOLoeSwJovQjdz?usp=sharing>`_ and
  `HalfCheetah <https://colab.research.google.com/drive/1bser52bpItzmlME00IA0bbmPdp1Xm0fy?usp=sharing>`_).


Installation
------------

EnvPool is currently hosted on `PyPI <https://pypi.org/project/envpool/>`_.
It requires Python >= 3.7.

You can install EnvPool with the following command:

.. code-block:: bash

    $ pip install envpool

After installation, open a Python console and type

::

    import envpool
    print(envpool.__version__)

If no error occurs, you have successfully installed EnvPool.

EnvPool is still under development; you can also check out the documents in
stable version through `envpool.readthedocs.io/en/stable/
<https://envpool.readthedocs.io/en/stable/>`_.


.. toctree::
   :maxdepth: 1
   :caption: Content

   content/slides
   content/build
   content/python_interface
   content/xla_interface
   content/benchmark
   content/new_env
   content/contributing


.. toctree::
   :maxdepth: 1
   :caption: Environment

   env/atari
   env/box2d
   env/classic_control
   env/dm_control
   env/minigrid
   env/mujoco_gym
   env/procgen
   env/toy_text
   env/vizdoom


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
