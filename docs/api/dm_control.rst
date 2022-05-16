DeepMind Control Suite
======================

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

BallInCupCatch-v1
-----------------

`dm_control suite ball-in-cup source code
<https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/ball_in_cup.py>`_

- Observation spec: a namedtuple with two keys: ``position (4)`` and
  ``velocity (4)``;
- Action spec: ``(2)``, with range ``[-1, 1]``;
- ``frame_skip``: 1;
- ``max_episode_steps``: 1000;

CheetahRun-v1
-------------

`dm_control suite cheetah source code
<https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/cheetah.py>`_

- Observation spec: a namedtuple with two keys: ``position (8)`` and
  ``velocity (9)``;
- Action spec: ``(6)``, with range ``[-1, 1]``;
- ``frame_skip``: 1;
- ``max_episode_steps``: 1000;


FingerSpin-v1, FingerTurnEasy-v1, FingerTurnHard-v1
---------------------------------------------------

`dm_control suite finger source code
<https://github.com/deepmind/dm_control/blob/main/dm_control/suite/finger.py>`_

- Observation spec: a namedtuple with five keys: ``position (4)``,
  ``velocity (3)``, ``touch (2)``, ``target_position (2)``,
  ``dist_to_target ()``;
- Action spec: ``(2)``, with range ``[-1, 1]``;
- ``frame_skip``: 2;
- ``max_episode_steps``: 1000;

.. note ::

    The observation keys ``target_position`` and ``dist_to_target`` are only
    available in ``FingerTurnEasy-v1`` and ``FingerTurnHard-v1`` tasks. Their
    values are meaningless in ``FingerSpin-v1``.


HopperStand-v1, HopperHop-v1
----------------------------

`dm_control suite hopper source code
<https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/hopper.py>`_

- Observation spec: a namedtuple with three keys: ``position (6)``,
  ``velocity (7)``, ``touch (2)``;
- Action spec: ``(4)``, with range ``[-1, 1]``;
- ``frame_skip``: 4;
- ``max_episode_steps``: 1000;


PendulumSwingup-v1
------------------

`dm_control suite pendulum source code
<https://github.com/deepmind/dm_control/blob/main/dm_control/suite/pendulum.py>`_

- Observation spec: a namedtuple with three keys: ``orientations (2)``,
  ``velocity (1)``;
- Action spec: ``(1)``, with range ``[-1, 1]``;
- ``frame_skip``: 1;
- ``max_episode_stes``: 1000;


ReacherEasy-v1, ReacherHard-v1
------------------------------

`dm_control suite reacher source code
<https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/reacher.py>`_

- Observation spec: a namedtuple with three keys: ``position (2)``,
  ``to_target (2)`` and ``velocity (2)``;
- Action spec: ``(2)``, with range ``[-1, 1]``;
- ``frame_skip``: 1;
- ``max_episode_steps``: 1000;


WalkerRun-v1, WalkerStand-v1, WalkerWalk-v1
-------------------------------------------

`dm_control suite walker source code
<https://github.com/deepmind/dm_control/blob/1.0.2/dm_control/suite/walker.py>`_

- Observation spec: a namedtuple with three keys: ``orientations (14)``,
  ``height ()`` and ``velocity (9)``;
- Action spec: ``(6)``, with range ``[-1, 1]``;
- ``frame_skip``: 10;
- ``max_episode_steps``: 1000;
