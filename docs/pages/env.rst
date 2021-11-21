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


CartPoleEnvSpec
~~~~~~~~~~~~~~~

Next, we need to create ``CartPoleEnvSpec`` to define the env-specific config,
state space, and action space. Create a class ``CartPoleEnvFns``:

.. code-block:: c++

    // env-specific definition of config and state/action spec
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
        // the last argument in Spec is for the range definition
        return MakeDict("action"_.bind(Spec<int>({-1}, {0, 1})));
      }
    };

    // this line will concat common config and common state/action spec
    typedef class EnvSpec<CartPoleEnvFns> CartPoleEnvSpec;

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

.. danger ::

    When using string in ``MakeDict``, you should explicit use ``std::string``.
    For example,

    .. code-block:: c++

        auto config = MakeDict("path"_.bind("init_path"));

    this will be a ``const char *`` type instead of ``std::string``, which will
    cause sometimes ``config["path"_]`` be a meaningless string in further
    usage. Instead, you should change the code as

    .. code-block:: c++

        auto config = MakeDict("path"_.bind(std::string("init_path")));

.. note ::

    ``-1`` in Spec is reserved for number of players. In single-player
    environment, ``Spec<int>({-1})`` is the same as ``Spec<int>({})`` (empty
    shape), but in multi-player environment, empty shape spec will be only a
    single int value per environment, while the former will be an array with
    length == #players (can be 0 when all players are dead).

.. note ::

    The common config and common state/action spec are defined in
    `env_spec.h <https://github.com/sail-sg/envpool/blob/master/envpool/core/env_spec.h>`_.

.. note ::

    EnvPool supports the environment that has multiple observations, or even
    nested observations. For example, ``FetchReach-v1``:

    ::

        >>> import gym
        >>> env = gym.make("FetchReach-v1")
        >>> e.observation_space
        Dict(achieved_goal:Box([-inf ...], [inf ...], (3,), float32), desired_goal:Box([-inf ...], [inf ...], (3,), float32), observation:Box([-inf ...], [inf ...], (10,), float32))
        >>> env.reset()
        >>> env.step([0, 0, 0, 0])
        ({'observation': array([ 1.34185919e+00,  7.49100661e-01,  5.34545376e-01,  0.00000000e+00,
                  0.00000000e+00,  2.49364315e-05,  2.35502607e-07, -1.56066826e-04,
                  3.22889321e-06, -1.55593223e-06]),
          'achieved_goal': array([1.34185919, 0.74910066, 0.53454538]),
          'desired_goal': array([1.36677977, 0.67090477, 0.60136475])},
         -1.0,
         False,
         {'is_success': 0.0})

    If we want to create such a state spec (including both obs and info), here
    is the solution:

    .. code-block:: c++

        template <typename Config>
        static decltype(auto) StateSpec(const Config& conf) {
          return MakeDict(
            "obs:observation"_.bind(Spec<float>({10})),
            "obs:achieved_goal"_.bind(Spec<float>({3})),
            "obs:desired_goal"_.bind(Spec<float>({3})),
            "info:is_success"_.bind(Spec<float>({})));
        }

    The keys start with ``obs:`` will be parsed to obs dict, and similarly
    ``info:`` will be parsed to info dict.

    For nested observations such as ``{"obs_a": {"obs_b": 6}}``, use ``.`` to
    indicate the hierarchy:

    .. code-block:: c++

        return MakeDict("obs:obs_a.obs_b"_.bind(Spec<int>({})));

.. note ::

    In dm_env, keys in Spec that start with either ``obs:`` or ``info:`` will
    be merged together under ``timestep.observation``.

CartPoleEnvPool
~~~~~~~~~~~~~~~
