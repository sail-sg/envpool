DeepMind Control Suite
======================

We use ``dm_control==1.0.3.post1`` and ``mujoco==2.2.0`` as the codebase.
See https://github.com/deepmind/dm_control/tree/1.0.3.post1 and
https://github.com/deepmind/mujoco/tree/2.2.0

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


AcrobotSwingup-v1, AcrobotSwingupSparse-v1
------------------------------------------

`dm_control suite acrobot source code
<https://github.com/deepmind/dm_control/blob/1.0.3.post1/dm_control/suite/acrobot.py>`_

- Observation spec: a namedtuple with two keys: ``orientations (4)``,
  ``velocity (2)``;
- Action spec: ``(1)``, with range ``[-1, 1]``;
- ``frame_skip``: 1;
- ``max_episode_stes``: 1000;


BallInCupCatch-v1
-----------------

`dm_control suite ball-in-cup source code
<https://github.com/deepmind/dm_control/blob/1.0.3.post1/dm_control/suite/ball_in_cup.py>`_

- Observation spec: a namedtuple with two keys: ``position (4)`` and
  ``velocity (4)``;
- Action spec: ``(2)``, with range ``[-1, 1]``;
- ``frame_skip``: 1;
- ``max_episode_steps``: 1000;


CartpoleBalance-v1, CartpoleBalanceSparse-v1, CarpoletSwingup-v1, CartpoleSwingupSparse-v1, CartpoleTwoPoles-v1, CartPoleThreePoles-v1
--------------------------------------------------------------------------------------------------------------------------------------

`dm_control suite cartpole source code
<https://github.com/deepmind/dm_control/blob/1.0.3.post1/dm_control/suite/cartpole.py>`_

- Observation spec: a namedtuple with two keys: ``position (5 for two_poles, 7
  for three_poles, 3 for others)``, ``velocity (3 for two_poles, 4 for
  three_poles, 2 for others)``;
- Action spec: ``(1)``, with range ``[-1, 1]``;
- ``frame_skip``: 1;
- ``max_episode_stes``: 1000;


CheetahRun-v1
-------------

`dm_control suite cheetah source code
<https://github.com/deepmind/dm_control/blob/1.0.3.post1/dm_control/suite/cheetah.py>`_

- Observation spec: a namedtuple with two keys: ``position (8)`` and
  ``velocity (9)``;
- Action spec: ``(6)``, with range ``[-1, 1]``;
- ``frame_skip``: 1;
- ``max_episode_steps``: 1000;


FingerSpin-v1, FingerTurnEasy-v1, FingerTurnHard-v1
---------------------------------------------------

`dm_control suite finger source code
<https://github.com/deepmind/dm_control/blob/1.0.3.post1/dm_control/suite/finger.py>`_

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


FishSwim-v1, FishUpright-v1
---------------------------

`dm_control suite fish source code
<https://github.com/deepmind/dm_control/blob/1.0.3.post1/dm_control/suite/fish.py>`_

- Observation spec: a namedtuple with four keys: ``joint_angles (7)``,
  ``upright ()``, ``target (3)``, ``velocity (13)``;
- Action spec: ``(5)``, with range ``[-1, 1]``;
- ``frame_skip``: 10;
- ``max_episode_steps``: 1000;

.. note ::

    The observation key ``target`` is only available in ``FishSwim-v1`` task.
    The value is meaningless in ``FishUpright-v1``.


HopperStand-v1, HopperHop-v1
----------------------------

`dm_control suite hopper source code
<https://github.com/deepmind/dm_control/blob/1.0.3.post1/dm_control/suite/hopper.py>`_

- Observation spec: a namedtuple with three keys: ``position (6)``,
  ``velocity (7)``, ``touch (2)``;
- Action spec: ``(4)``, with range ``[-1, 1]``;
- ``frame_skip``: 4;
- ``max_episode_steps``: 1000;


ManipulatorBringBall-v1, ManipulatorBringPeg-v1, ManipulatorInsertBall-v1, ManipulatorInsertPeg-v1
--------------------------------------------------------------------------------------------------

`dm_control suite manipulator source code
<https://github.com/deepmind/dm_control/blob/1.0.3.post1/dm_control/suite/manipulator.py>`_

- Observation spec: a namedtuple with three keys: ``arm_pos (8,2)``,
  ``arm_vel (8)``, ``touch (5)``, ``hand_pos (4)``, ``object_pos (4)``,
  ``object_vel (3)``, ``target_pos (4)``;
- Action spec: ``(5)``, with range ``[-1, 1]``;
- ``frame_skip``: 10;
- ``max_episode_steps``: 1000;


HumanoidStand-v1, HumanoidWalk-v1, HumanoidRun-v1, HumanoidRunPureState-v1
--------------------------------------------------------------------------

`dm_control suite humanoid source code
<https://github.com/deepmind/dm_control/blob/1.0.3.post1/dm_control/suite/humanoid.py>`_

- Observation spec: a namedtuple with seven keys: ``joint_angles (21)``,
  ``head_height ()``, ``extremities (12)``, ``torso_vertical (3)``,
  ``com_velocity (3)``, ``position (28)``, and ``velocity (27)``;
- Action spec: ``(21)``, with range ``[-1, 1]``;
- ``frame_skip``: 5;
- ``max_episode_steps``: 1000;

.. note ::

    The observation keys ``joint_angles``, ``head_height``, ``extremities``,
    ``torso_vertical`` and ``com_velocity`` are only available in
    ``HumanoidStand-v1``, ``HumanoidWalk-v1`` and ``HumanoidRun-v1``.
    The observation keys ``position`` are only available in
    ``HumanoidRunPureState-v1`` tasks.


HumanoidCMUStand-v1, HumanoidCMURun-v1
--------------------------------------

`dm_control suite humanoid-CMU source code
<https://github.com/deepmind/dm_control/blob/1.0.3.post1/dm_control/suite/humanoid_CMU.py>`_

- Observation spec: a namedtuple with six keys: ``joint_angles (56)``,
  ``head_height ()``, ``extremities (12)``, ``torso_vertical (3)``,
  ``com_velocity (3)`` and ``velocity (62)``;
- Action spec: ``(56)``, with range ``[-1, 1]``;
- ``frame_skip``: 10;
- ``max_episode_steps``: 1000;


PendulumSwingup-v1
------------------

`dm_control suite pendulum source code
<https://github.com/deepmind/dm_control/blob/1.0.3.post1/dm_control/suite/pendulum.py>`_

- Observation spec: a namedtuple with three keys: ``orientations (2)``,
  ``velocity (1)``;
- Action spec: ``(1)``, with range ``[-1, 1]``;
- ``frame_skip``: 1;
- ``max_episode_stes``: 1000;


PointMassEasy-v1, PointMassHard-v1
----------------------------------

`dm_control suite point-mass source code
<https://github.com/deepmind/dm_control/blob/1.0.3.post1/dm_control/suite/point_mass.py>`_

- Observation spec: a namedtuple with three keys: ``position (2)``,
  ``velocity (2)``;
- Action spec: ``(1)``, with range ``[-1, 1]``;
- ``frame_skip``: 1;
- ``max_episode_stes``: 1000;


ReacherEasy-v1, ReacherHard-v1
------------------------------

`dm_control suite reacher source code
<https://github.com/deepmind/dm_control/blob/1.0.3.post1/dm_control/suite/reacher.py>`_

- Observation spec: a namedtuple with three keys: ``position (2)``,
  ``to_target (2)`` and ``velocity (2)``;
- Action spec: ``(2)``, with range ``[-1, 1]``;
- ``frame_skip``: 1;
- ``max_episode_steps``: 1000;


SwimmerSwimmer6-v1, SwimmerSwimmer15-v1
---------------------------------------

`dm_control suite swimmer source code
<https://github.com/deepmind/dm_control/blob/1.0.3.post1/dm_control/suite/swimmer.py>`_

- Observation spec: a namedtuple with three keys: ``joints (5 for swimmer6,
  14 for swimmer15)``, ``to_target (2)``, and ``body_velocities (18 for
  swimmer6, 45 for swimmer15)``;
- Action spec: ``(5 for swimmer6, 14 for swimmer15)``, with range ``[-1, 1]``;
- ``frame_skip``: 15;
- ``max_episode_steps``: 1000;


WalkerRun-v1, WalkerStand-v1, WalkerWalk-v1
-------------------------------------------

`dm_control suite walker source code
<https://github.com/deepmind/dm_control/blob/1.0.3.post1/dm_control/suite/walker.py>`_

- Observation spec: a namedtuple with three keys: ``orientations (14)``,
  ``height ()`` and ``velocity (9)``;
- Action spec: ``(6)``, with range ``[-1, 1]``;
- ``frame_skip``: 10;
- ``max_episode_steps``: 1000;
