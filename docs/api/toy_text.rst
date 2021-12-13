Toy Text
========

Most of the environments in classic control borrow from `Gym
<https://github.com/openai/gym/tree/master/gym/envs/toy_text>`_ and
`bsuite <https://github.com/deepmind/bsuite/tree/master/bsuite/environments>`_.


Catch-v0
--------

`bsuite catch source code
<https://github.com/deepmind/bsuite/blob/master/bsuite/environments/catch.py>`_

The agent must move a paddle to intercept falling balls. Falling balls only
move downwards on the column they are in.


FrozenLake-v1, FrozenLake8x8-v1
-------------------------------

`gym FrozenLake-v1 source code
<https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py>`_

The agent controls the movement of a character in a grid world. Some tiles of
the grid are walkable, and others lead to the agent falling into the water.
Additionally, the movement direction of the agent is uncertain and only
partially depends on the chosen direction. The agent is rewarded for finding a
walkable path to a goal tile.

The difference between ``FrozenLake-v1`` and ``FrozenLake8x8-v1`` is that the
former has ``size`` 4, ``max_episode_steps`` 100 with 0.7 reward threshold,
while the latter has ``size`` 8, ``max_episode_steps`` 200 with 0.85 reward
threshold.
