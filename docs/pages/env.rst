Add New Environment into EnvPool
================================

To add a new environment in C++ that EnvPool will parallelly run, we provide a
developer interface in `envpool/core/env.h
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

Here is the expected file structure:

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


Implement CartPole Environment in cartpole.h
--------------------------------------------

First, include the core header files:

.. code-block:: c++

    #include "envpool/core/async_envpool.h"
    #include "envpool/core/env.h"


CartPoleEnvSpec
~~~~~~~~~~~~~~~

Next, we need create ``CartPoleEnvSpec`` to define the env-specific config,
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

CartPole is quite a simple environment. The observation is a NumPy array with
shape ``(4,)``, and the action is discrete ``[0, 1]``. This definition is also
available to see on the python side:

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

    When using a string in ``MakeDict``, you should explicitly use
    ``std::string``. For example,

    .. code-block:: c++

        auto config = MakeDict("path"_.bind("init_path"));

    This will be a ``const char *`` type instead of ``std::string``, which will
    sometimes cause ``config["path"_]`` to be a meaningless string in further
    usage. Instead, you should change the code as

    .. code-block:: c++

        auto config = MakeDict("path"_.bind(std::string("init_path")));

.. note ::

    The above example shows how to define a discrete action space by specifying
    the last argument of ``Spec``. Here is another example, if our environment
    has 6 actions, ranging from 0 to 5:

    .. code-block:: c++

        template <typename Config>
        static decltype(auto) ActionSpec(const Config& conf) {
          return MakeDict("action"_.bind(Spec<int>({-1}, {0, 5})));
          // or remove -1, no difference in single-player env
          // return MakeDict("action"_.bind(Spec<int>({}, {0, 5})));
        }

    For continuous action space, simply change the type of ``Spec`` to float or
    double. For example, if the action is a NumPy array with 4 floats, ranging
    from -2 to 2:

    .. code-block:: c++

        template <typename Config>
        static decltype(auto) ActionSpec(const Config& conf) {
          return MakeDict("action"_.bind(Spec<float>({-1, 4}, {-2.0f, 2.0f})));
          // or remove -1, no difference in single-player env
          // return MakeDict("action"_.bind(Spec<float>({4}, {-2.0f, 2.0f})));
        }

.. note ::

    ``-1`` in Spec is reserved for the number of players. In single-player
    environment, ``Spec<int>({-1})`` is the same as ``Spec<int>({})`` (empty
    shape), but in a multi-player environment, empty shape spec will be only a
    single int value per environment, while the former will be an array with
    length == #players (can be 0 when all players are dead).

.. note ::

    The common config and common state/action spec are defined in
    `env_spec.h <https://github.com/sail-sg/envpool/blob/master/envpool/core/env_spec.h>`_.

.. note ::

    EnvPool supports the environment that has multiple observations or even
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

    It is the same as ActionSpec. The only difference is: there's no ``obs:``
    and ``info:`` in action.

.. note ::

    In dm_env, keys in Spec that start with either ``obs:`` or ``info:`` will
    be merged under ``timestep.observation``.


CartPoleEnv
~~~~~~~~~~~

Now we are going to create a class ``CartPoleEnv`` that inherits
`Env <https://github.com/sail-sg/envpool/blob/master/envpool/core/env.h>`_.

We have already defined three types ``Spec``, ``State`` and ``Action`` in Env
class for convenience, which follow the definition of ``CartPoleEnvSpec``.

The following functions are required to override:

- constructor, in this case it is ``CartPoleEnv(const Spec& spec, int env_id)``;
  you can use ``spec.config["max_episode_steps"_]`` to extract the value from
  config;
- ``bool IsDone()``: return a boolean that indicate whether the current episode
  is finished or not;
- ``void Reset()``: perform one ``env.reset()``;
- ``void Step(const Action& action)``: perform one ``env.step(action)``.


Array Read/Write
~~~~~~~~~~~~~~~~

``State`` and ``Action`` are dict-style data structures for easier prototyping.
All values in these dictionaries are with type ``Array``, which mimic the
functionality of a multi-dimensional array.

To extract value from action in ``CartPoleEnv``:

.. code-block:: c++

    // auto convert the first element in action["action"_]
    int act = action["action"_];
    // for continuous action space, e.g.
    // float act2 = action["action"_][2];

If the state/action contains several keys and each element is a
multi-dimensional array, e.g., an image, there are three ways to deal with array
read/write:

.. code-block:: c++

    uint8_t *ptr = static_cast<uint8_t *>(state["obs"_].data());

    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 84; ++j) {
        for (int k = 0; k < 84; ++k) {
          // 1. use []
          state["obs"_][i][j][k] = ...
          // 2. use (), faster than 1
          state["obs"_](i, j, k) = ...
          // 3. use raw pointer
          ptr[i * 84 * 84 + j * 84 + k] = ...
        }
      }
    }


Allocate State in Reset and Step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EnvPool has carefully designed the data movement to achieve zero-copy
with the lowest overhead. We create a simple API to make it be more
user-friendly.

At the end of ``Reset`` and ``Step`` function, you need to call ``Allocate``
method to allocate state for writing. For example, in CartPoleEnv:

.. code-block:: c++

    State state = Allocate();
    state["obs"_][0] = static_cast<float>(x_);
    state["obs"_][1] = static_cast<float>(x_dot_);
    state["obs"_][2] = static_cast<float>(theta_);
    state["obs"_][3] = static_cast<float>(theta_dot_);
    state["reward"_] = 1.0f;

    // here is a buggy usage because x_ is float64 and state["obs"_] is float32
    // state["obs"_][0] = x_;


You do not pass this state to any other functions or return. Instead,
AsyncEnvPool will automatically process the data and pack it to the python
interface.

.. note ::

    For multi-player environments, you need to allocate state with an extra
    argument ``player_num``. For example, if the state spec is:

    .. code-block:: c++

        template <typename Config>
        static decltype(auto) StateSpec(const Config& conf) {
          return MakeDict(
            "obs:players.obs"_.bind(Spec<uint8_t>({-1, 4, 84, 84})),
            "obs:players.location"_.bind(Spec<uint8_t>({-1, 2})),
            "info:players.health"_.bind(Spec<int>({-1})),
            "info:player_num"_.bind(Spec<int>({})),
            "info:bla"_.bind(Spec<float>({2, 3, 3}))
          );
        }

    By calling ``auto state = Allocate(10)``, the state would be like:

    .. code-block:: c++

        state["obs:players.obs"_];      // shape: (10, 4, 84, 84)
        state["obs:players.location"];  // shape: (10, 2)
        state["info:players.health"];   // shape: (10,)
        state["info:player_num"];       // shape: (), only one element
        state["info:bla"];              // shape: (2, 3, 3)

.. danger ::

    Please make sure the types are correct. Assigning int to a float array or
    assigning double to an uint64_t array will not generate any compilation
    error, but in the actual runtime, the data is wrong. Please use
    ``static_cast`` to convert the type correctly.


CartPoleEnvPool
~~~~~~~~~~~~~~~

After creating ``CartPoleEnv``, just one more line we can get
``CartPoleEnvPool``:

.. code-block:: c++

    typedef AsyncEnvPool<CartPoleEnv> CartPoleEnvPool;


Miscellaneous
~~~~~~~~~~~~~

.. note ::

    Please do not use the pseudo-random number by ``rand() % MAX``. Instead,
    use `random number distributions
    <https://en.cppreference.com/w/cpp/numeric/random>`_ to generate
    thread-safe deterministic pseudo-random number. ``std::mt19937`` generator
    has already been defined as ``gen_`` (`link
    <https://github.com/sail-sg/envpool/blob/v0.4.0/envpool/core/env.h#L37>`_).


Generate Dynamic Linked .so File by classic_control.cc
------------------------------------------------------

We use `pybind11 <https://github.com/pybind/pybind11>`_ to let python interface
use this C++ code. We have already wrapped this interface, you just need to add
only a few lines to make it work:

.. code-block:: c++

    #include "envpool/classic_control/cartpole.h"
    #include "envpool/core/py_envpool.h"

    // generate python-side CartPoleEnvSpec
    typedef PyEnvSpec<classic_control::CartPoleEnvSpec> CartPoleEnvSpec;
    // generate python-side CartPoleEnvPool
    typedef PyEnvPool<classic_control::CartPoleEnvPool> CartPoleEnvPool;

    // generate classic_control_envpool.so
    PYBIND11_MODULE(classic_control_envpool, m) {
      REGISTER(m, CartPoleEnvSpec, CartPoleEnvPool)
    }


Write Bazel BUILD File
----------------------


Register CartPole-v0/1 in EnvPool
---------------------------------


Add Unit Test for CartPoleEnv
-----------------------------
