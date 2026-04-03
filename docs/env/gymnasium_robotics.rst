Gymnasium-Robotics
==================

EnvPool provides native C++ backends for the Gymnasium-Robotics Fetch, Shadow
Hand, Adroit, PointMaze, and FrankaKitchen task families. The Python package
exposes the same ``make_dm``, ``make_gym``, and ``make_gymnasium`` entry points
as other EnvPool environments, including batched sync and async stepping.

Supported Tasks
---------------

EnvPool mirrors Adroit, Fetch, Shadow Hand manipulation/reach, PointMaze, and
FrankaKitchen task IDs from Gymnasium-Robotics. Legacy MuJoCo ``*-v2`` IDs such
as ``Ant-v2`` and ``HalfCheetah-v2`` are intentionally not registered, because
EnvPool already provides the corresponding native MuJoCo
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

``FrankaKitchen-v1`` uses a fixed all-task observation schema in EnvPool so
that the C++ state specification remains static. The returned ``info`` values
for ``tasks_to_complete``, ``step_task_completions``, and
``episode_task_completions`` are 7-dimensional masks ordered as
``bottom burner``, ``top burner``, ``light switch``, ``slide cabinet``,
``hinge cabinet``, ``microwave``, and ``kettle``.
