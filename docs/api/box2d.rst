Box2D
=====

We use ``box2d==2.4.1`` and ``gym==0.23.1`` as the codebase. See
https://github.com/erincatto/box2d/tree/v2.4.1 and
https://github.com/openai/gym/tree/v0.23.1/gym/envs/box2d


CarRacing-v1
------------

The easiest control task to learn from pixels - a top-down racing environment.
The generated track is random every episode.

Action Space
~~~~~~~~~~~~

There are 3 actions: steering (-1 for full left, 1 for full right), gas
(0 ~ 1), and breaking (0 ~ 1).

Observation Space
~~~~~~~~~~~~~~~~~

State consists of 3 channel 96x96 pixels.

Rewards
~~~~~~~

The reward is -0.1 every frame and +1000/N for every track tile visited, where
N is the total number of tiles visited in the track. For example, if you have
finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

Starting State
~~~~~~~~~~~~~~

The car starts at rest in the center of the road.

Episode Termination
~~~~~~~~~~~~~~~~~~~

The episode finishes when all of the tiles are visited. The car can also go
outside of the playfield - that is, far off the track, in which case it will
receive -100 reward and die.
