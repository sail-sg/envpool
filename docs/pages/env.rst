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
<https://github.com/sail-sg/envpool/pull/25/files>`_.
Let's go through the details step by step!


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


Implement CartPole environment in cartpole.h
--------------------------------------------

First, include the core header files:

.. code-block:: c++

    #include "envpool/core/async_envpool.h"
    #include "envpool/core/env.h"

Next, we need to create ``CartPoleEnvSpec`` to define the env-specific config,
state space, and action space. We first create a class ``CartPoleEnvFns``:

.. code-block:: c++

    class CartPoleEnvFns {
     public:
      static decltype(auto) DefaultConfig() {
        return MakeDict("max_episode_steps"_.bind(200),
                        "reward_threshold"_.bind(195.0));
      }
      template <typename Config>
      static decltype(auto) StateSpec(const Config& conf) {
        return MakeDict("obs"_.bind(Spec<float>({4})));
      }
      template <typename Config>
      static decltype(auto) ActionSpec(const Config& conf) {
        return MakeDict("action"_.bind(Spec<int>({-1}, {0, 1})));
      }
    };

- ``DefaultConfig``: the default config to create cartpole environment;
- ``StateSpec``: the state space (including observation and info) definition;
- ``ActionSpec``: the action space definition.

CartPole is quite a simple environment. The observation is a numpy array with
shape ``(4,)``, and the action is a discrete action ``[0, 1]``. This
definition is also available to see in python side:

::

    >>> import envpool
    >>> spec = envpool.make_spec("CartPole-v0")
    >>> spec
    CartPoleEnvSpec(num_envs=1, batch_size=0, num_threads=0, max_num_players=1, thread_affinity_offset=-1, base_path='envpool', seed=42, max_episode_steps=200, reward_threshold=195.0)

    >>> # if we change a config value
    >>> env = envpool.make_gym("CartPole-v0", reward_threshold=666)
    >>> env
    CartPoleGymEnvPool(num_envs=1, batch_size=0, num_threads=0, max_num_players=1, thread_affinity_offset=-1, base_path='envpool', seed=42, max_episode_steps=200, reward_threshold=666.0)

    >>> # observation space and action space
    >>> env.observation_space
    Box([1.1754944e-38 1.1754944e-38 1.1754944e-38 1.1754944e-38], [3.4028235e+38 3.4028235e+38 3.4028235e+38 3.4028235e+38], (4,), float32)
    >>> env.action_space
    Discrete(2)
    >>> env.spec.reward_threshold
    666.0

