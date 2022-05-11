dm_control suite benchmark
==========================

We use ``dm_control==1.0.2`` and ``mujoco==2.1.5`` as the codebase.
See https://github.com/deepmind/dm_control/tree/1.0.2 and
https://github.com/deepmind/mujoco/tree/2.1.5

The ``domain_name`` and ``task_name`` for ``suite.load`` function are
converted into ``DomainNameTaskName-v1`` in envpool, e.g.,

::

  dm_raw_env = suite.load("hopper", "hop")
  # equal to
  envpool_env = envpool.make_dm("HopperHop-v1", num_envs=1)

  # if _ is in the original name
  suite.load("ball_in_cup", "catch")
  # equal to
  envpool.make_dm("BallInCupCatch-v1", num_envs=1)


HopperStand-v1, HopperHop-v1
----------------------------

`dm_control suite hopper source code
<https://github.com/deepmind/dm_control/blob/main/dm_control/suite/hopper.py>`_

- Observation spec: a namedtuple with three keys: ``position (6)``,
  ``velocity (7)``, ``touch (2)``;
- Action spec: ``(4)``, with range ``[-1, 1]``;
- ``frame_skip``: 4;
- ``max_episode_steps``: 1000;

