Gymnasium-Robotics
==================

EnvPool registers upstream ``gymnasium_robotics==1.4.2`` task IDs when the
optional ``gymnasium-robotics`` package is installed. This backend wraps one
upstream Gymnasium-Robotics environment per EnvPool slot and focuses on API and
registry compatibility for ``make_dm``, ``make_gym``, and ``make_gymnasium``.

Supported Tasks
---------------

EnvPool mirrors Adroit, Fetch, Shadow Hand manipulation/reach, AntMaze,
PointMaze, and FrankaKitchen task IDs from Gymnasium-Robotics. Legacy MuJoCo
``*-v2`` IDs such as ``Ant-v2`` and ``HalfCheetah-v2`` are intentionally not
registered, because EnvPool already provides the corresponding native MuJoCo
``*-v3``/``*-v4``/``*-v5`` tasks.

For legacy Gymnasium-Robotics IDs that still point to the deprecated
``mujoco_py`` backend upstream, EnvPool transparently routes to the modern
MuJoCo equivalents:

* ``Fetch*`` ``v1`` IDs use the corresponding ``v4`` tasks.
* ``HandManipulate*`` ``v0`` IDs use the corresponding ``v1`` tasks.
* ``HandReach-v0`` and ``HandReachDense-v0`` use the corresponding ``v3``
  tasks.

Examples
--------

.. code-block:: python

    import envpool

    env = envpool.make_gymnasium("FetchReach-v4", num_envs=4, seed=0)
    obs, info = env.reset()
    obs, rew, term, trunc, info = env.step(env.action_space.sample()[None, :].repeat(4, axis=0))
    frame = env.render()
    env.close()

Notes
-----

This backend currently runs in synchronous mode only. Use ``batch_size=0`` or
``batch_size=num_envs`` when creating Gymnasium-Robotics EnvPools.
