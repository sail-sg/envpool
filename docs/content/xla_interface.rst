XLA Interface
=============

To boost the efficiency of the overall system, we introduce the XLA API for envpool.
With this API, we can just-in-time compile the environment and agent steps together,
when the agent part is implemented with Jax/Tensorflow.

The full example is at https://github.com/sail-sg/envpool/blob/main/examples/xla_step.py


Stateless functions
-------------------

The main issue with jitting the environment is that the ``env.step(action) -> state``
(and similarly the ``recv/send``) function is not a pure function,
i.e. it changes the state of the underlying ``env``.
To overcome this issue, we introduce a pure functional version of ``step`` (``recv/send``).

Namely, the XLA version of ``step/recv/send`` has the follow signature:
::

    step(envpool_handle: Handle, action: Action) -> Tuple[Handle, State]
    recv(envpool_handle: Handle) -> Tuple[Handle, State]
    send(envpool_handle: Handle, action: Action) -> Handle

These functions can be obtained from the envpool instance which we created
from the Python API.
::

    env = envpool.make(..., env_type="gym" | "dm")
    handle, recv, send, step = env.xla()


Example of Actor Loop
---------------------

We can now write the actor loop as:
::

    def actor_step(iter, loop_var):
      handle0, states = loop_var
      action = policy(states)
      # for gym
      handle1, (new_states, rew, done, info) = step(handle0, action)
      # for dm
      # handle1, new_states = step(handle0, action)
      return (handle1, new_states)

    @jit
    def run_actor_loop(num_steps, init_var):
      return lax.fori_loop(0, num_steps, actor_step, init_var)

    states = env.reset()
    run_actor_loop(100, (handle, states))

Or, with the asynchronous api:
::

    def actor_step(iter, handle):
      handle0 = handle
      handle1, states = recv(handle0)
      action = policy(states.observation.obs)
      handle2 = send(handle0, action, states.observation.env_id)
      return handle2

    @jit
    def run_actor_loop(num_steps):
      return lax.fori_loop(0, num_steps, actor_step, handle)

    env.async_reset()
    run_actor_loop(100)

It is also possible to overlap ``send`` and ``recv``:
::

    def actor_step(iter, loop_var):
      handle0, states = loop_var
      action = policy(states.observation.obs)
      handle1 = send(handle0, action, states.observation.env_id)
      handle1, new_states = recv(handle0)
      return handle1, new_states

    @jit
    def run_actor_loop(num_steps, init_var):
      return lax.fori_loop(0, num_steps, actor_step, init_var)

    env.async_reset()
    handle, states = recv(handle)
    run_actor_loop(100, (handle, states))

In the above case, ``recv`` is using ``handle0``, which means ``policy`` and
``recv`` will be overlapped in each iteration.
