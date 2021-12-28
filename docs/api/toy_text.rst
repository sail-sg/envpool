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


Taxi-v3
-------

`gym Taxi-v3 source code
<https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py>`_

There are four designated locations in the grid world indicated by Red, Green,
Yellow, and Blue. When the episode starts, the taxi starts off at a random
square and the passenger is at a random location. The taxi drives to the
passenger's location, picks up the passenger, drives to the passenger's
destination (another one of the four specified locations), and then drops off
the passenger. Once the passenger is dropped off, the episode ends.


NChain-v0
---------

`gym NChain-v0 source code
<https://github.com/openai/gym/blob/v0.20.0/gym/envs/toy_text/nchain.py>`_

This game presents moves along a linear chain of states, with two actions:

0. forward, which moves along the chain but returns no reward
1. backward, which returns to the beginning and has a small reward

The end of the chain, however, presents a large reward, and by moving
'forward' at the end of the chain this large reward can be repeated.
At each action, there is a small probability that the agent 'slips' and the
opposite transition is instead taken.
The observed state is the current state in the chain (0 to n-1).


CliffWalking-v0
---------------

`gym CliffWalking-v0 source code
<https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py>`_

The board is a 4x12 matrix, with (using NumPy matrix indexing):

- [3, 0] as the start at bottom-left
- [3, 11] as the goal at bottom-right
- [3, 1..10] as the cliff at bottom-center

Each time step incurs -1 reward, and stepping into the cliff incurs -100
reward and a reset to the start. An episode terminates when the agent reaches
the goal.


Blackjack-v1
------------

`gym Blackjack-v1 source code
<https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py>`_

Blackjack is a card game where the goal is to obtain cards that sum to as near
as possible to 21 without going over. They're playing against a fixed dealer.
Face cards (Jack, Queen, King) have point value 10. Aces can either count as
11 or 1, and it's called 'usable' at 11.

This game is placed with an infinite deck (or with replacement). The game
starts with dealer having one face up and one face down card, while player
having two face up cards. (Virtually for all Blackjack games today). The player
can request additional cards (hit=1) until they decide to stop (stick=0) or
exceed 21 (bust). After the player sticks, the dealer reveals their facedown
card, and draws until their sum is 17 or greater. If the dealer goes bust the
player wins. If neither player nor dealer busts, the outcome (win, lose, draw)
is decided by whose sum is closer to 21. The reward for winning is +1, drawing
is 0, and losing is -1.
