Mujoco
======

We use ``mujoco==2.1.5`` as the codebase.
See https://github.com/deepmind/mujoco/tree/2.1.5

The implementation follows OpenAI gym \*-v4 environment, see
`reference <https://github.com/openai/gym/tree/master/gym/envs/mujoco>`_.

You can set ``post_constraint`` to ``False`` to disable the bug fix with
`this issue <https://github.com/openai/gym/issues/2593>`_, which is \*-v3
environments' standard approach.


Ant-v3/v4
---------

`gym Ant-v3 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant_v3.py>`_

`gym Ant-v4 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant_v4.py>`_

- Observation space: ``(111)``, first 13 elements for ``qpos[2:]``, next 14
  elements for ``qvel``, other elements for clipped ``cfrc_ext`` (com-based
  external force on body, a.k.a. contact force);
- Action space: ``(8)``, with range ``[-1, 1]``.


HalfCheetah-v3/v4
-----------------

`gym HalfCheetah-v3 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/half_cheetah_v3.py>`_

`gym HalfCheetah-v4 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/half_cheetah_v4.py>`_

- Observation space: ``(17)``, first 8 elements for ``qpos[1:]``, next 9
  elements for ``qvel``;
- Action space: ``(6)``, with range ``[-1, 1]``.


Hopper-v3/v4
------------

`gym Hopper-v3 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/hopper_v3.py>`_

`gym Hopper-v4 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/hopper_v4.py>`_

- Observation space: ``(11)``, first 5 elements for ``qpos[1:]``, next 6
  elements for ``qvel``;
- Action space: ``(3)``, with range ``[-1, 1]``.


Humanoid-v3/v4, HumanoidStandup-v2/v4
-------------------------------------

`gym Humanoid-v3 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoid_v3.py>`_

`gym Humanoid-v4 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoid_v4.py>`_

`gym HumanoidStandup-v2 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoidstandup.py>`_

`gym HumanoidStandup-v4 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoidstandup_v4.py>`_

- Observation space: ``(376)``, first 22 elements for ``qpos[2:]``, next 23
  elements for ``qvel``, next 140 elements for ``cinert`` (com-based body
  inertia and mass), next 84 elements for ``cvel`` (com-based velocity [3D
  rot; 3D tran]), next 23 elements for ``qfrc_actuator`` (actuator force),
  next 84 elements for ``cfrc_ext`` (com-based external force on body);
- Action space: ``(17)``, with range ``[-0.4, 0.4]``.


InvertedPendulum-v2/v4
----------------------

`gym InvertedPendulum-v2 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/inverted_pendulum.py>`_

`gym InvertedPendulum-v4 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/inverted_pendulum_v4.py>`_

- Observation space: ``(4)``, first 2 elements for ``qpos``, next 2 elements
  for ``qvel``;
- Action space: ``(1)``, with range ``[-3, 3]``.


Pusher-v2/v4
------------

`gym Pusher-v2 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/pusher.py>`_

`gym Pusher-v4 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/pusher_v4.py>`_

- Observation space: ``(23)``, first 7 elements for ``qpos[:7]``, next 7
  elements for ``qvel[:7]``, next 3 elements for ``tips_arm``, next 3
  elements for ``object``, next 3 elements for ``goal``;
- Action space: ``(7)``, with range ``[-2, 2]``.


Reacher-v2/v4
-------------

`gym Reacher-v2 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/reacher.py>`_

`gym Reacher-v4 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/reacher_v4.py>`_

- Observation space: ``(11)``, first 2 elements for ``cos(qpos[:2])``, next 2
  elements for ``sin(qpos[:2])``, next 2 elements for ``qpos[2:]``, next 2
  elements for ``qvel[:2]``, next 3 elements for ``dist``, a.k.a.
  ``fingertip - target``;
- Action space: ``(2)``, with range ``[-1, 1]``.


Swimmer-v3/v4
-------------

`gym Swimmer-v3 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/swimmer_v3.py>`_

`gym Swimmer-v4 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/swimmer_v4.py>`_

- Observation space: ``(8)``, first 3 elements for ``qpos[2:]``, next 5
  elements for ``qvel``;
- Action space: ``(2)``, with range ``[-1, 1]``.
