.. EnvPool documentation master file, created by
   sphinx-quickstart on Mon Oct 25 21:01:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EnvPool!
===================

**EnvPool** is a C++-based batched environment pool with pybind11 and thread
pool. It has high performance (~1M raw FPS in DGX on Atari games) and
compatible APIs (supports both gym and dm_env, both sync and async, both single
and multi player environment).

Here are EnvPool's several highlights:

- Compatible with OpenAI ``gym`` APIs and DeepMind ``dm_env`` APIs;
- Manage a pool of envs, interact with the envs in batched APIs by default;
- Support both synchronous execution and asynchronous execution;
- Support both single player and multi-player environment;
- Easy C++ developer API to add new envs;
- **1 Million** Atari frames per second simulation with 256 CPU cores,
  **~13x** throughput of Python subprocess-based vector env;
- **~3x** throughput of Python subprocess-based vector env on low resource
  setup like 12 CPU cores;
- Comparing with the existing GPU-based solution
  (`Brax <https://github.com/google/brax>`_ /
  `Isaac-gym <https://developer.nvidia.com/isaac-gym>`_), EnvPool is a
  **general** solution for various kinds of speeding-up RL environment
  parallelization;
- Compatible with some existing RL libraries, e.g.,
  `Tianshou <https://github.com/thu-ml/tianshou>`_.

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
   :caption: Contents

   pages/slides
   pages/build
   pages/interface
   pages/benchmark
   pages/env
   pages/contributing


.. toctree::
   :maxdepth: 1
   :caption: Environments

   api/atari
   api/box2d
   api/classic
   api/dm_control
   api/mujoco
   api/toy_text
   api/vizdoom


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
