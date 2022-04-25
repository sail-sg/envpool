Mujoco
======

We use ``mujoco==2.1.5`` as the codebase.
See https://github.com/deepmind/mujoco/tree/2.1.5

The implementation follows OpenAI gym \*-v4 environment, see
`reference <https://github.com/openai/gym/tree/master/gym/envs/mujoco>`_.

You can set ``post_constraint`` to ``False`` to disable the bug fix with
`this issue <https://github.com/openai/gym/issues/2593>`_, which is \*-v3
environments' standard approach.


Ant-v4
------

`gym Ant-v4 source code
<https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant_v4.py>`_

- Observation space: ``(111)``, first 13 elements for ``qpos[2:]``, next 14
  elements for ``qvel``, other elements for clipped ``cfrc_ext`` (contact
  force);
- Action space: ``(8)``, with range ``[-1, 1]``.
