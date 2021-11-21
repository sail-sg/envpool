Add New Environment into EnvPool
================================

To add a new environment in C++ that will be parallelly run by EnvPool,
we provide a developer interface in `envpool/core/env.h
<https://github.com/sail-sg/envpool/blob/master/envpool/core/env.h>`_.

- For a quick and annotated example, please refer to
  `envpool/dummy/ <https://github.com/sail-sg/envpool/tree/master/envpool/dummy>`_.
- `envpool/atari
  <https://github.com/sail-sg/envpool/tree/master/envpool/atari>`_ serves as
  a more complex, real example.

In the following example, we will create an environment ``CartPole``.
It is the same version as `OpenAI gym
<https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py>`_.
The full implementation is in `Pull Request 25
<https://github.com/sail-sg/envpool/pull/25/files>`_, and let's go through
the details step by step!


Setup File Structure
--------------------

The first thing is to create a ``classic_control`` folder under ``envpool/``:

.. code-block:: bash

    cd envpool
    mkdir -p classic_control

Here are the expected file structure:

.. code-block:: bash

    $ tree classic_control
    classic_control
    ├── BUILD
    ├── cartpole.h
    ├── classic_control.cc
    ├── classic_control_test.py
    ├── __init__.py
    └── registration.py

and their functionalities:

- ``__init__.py``: to make this directory a python package;
- ``BUILD``: to indicate the file dependency (because we use bazel to manage
  this project);
- ``cartpole.h``: the CartPole environment;
- ``classic_control.cc``: pack ``classic_control_envpool.so`` via `pybind11
  <https://github.com/pybind/pybind11>`_;
- ``classic_control_test.py``: a simple unit-test to check if we implement
  correctly;
- ``registration.py``: register ``CartPole-v0`` and ``CartPole-v1`` so that
  we can use ``envpool.make("CartPole-v0")`` to create an environment.
