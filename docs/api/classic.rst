Classic Control
===============

All of the environments in classic control borrow from `Gym
<https://github.com/openai/gym/tree/master/gym/envs/classic_control>`_.


CartPole-v0/1
-------------

`gym CartPole-v0 source code
<https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py>`_

A pole is attached by an un-actuated joint to a cart, which moves along a
frictionless track. The pendulum starts upright, and the goal is to prevent it
from falling over by increasing and reducing the cart's velocity.

The difference between ``CartPole-v0`` and ``CartPole-v1`` is that the former
has ``max_episode_steps`` 200 with 195 reward threshold, while the latter has
``max_episode_steps`` 500 with 475 reward threshold.


Pendulum-v0
-----------

`gym Pendulum-v0 source code
<https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py>`_

The inverted pendulum swing-up problem is a classic problem in the control
literature. In this version of the problem, the pendulum starts in a random
position, and the goal is to swing it up to stay upright.


MountainCar-v0, MountainCarContinuous-v0
----------------------------------------

`gym MountainCar-v0 source code
<https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py>`_
and `gym MountainCarContinuous-v0 source code
<https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py>`_

The agent (a car) is started at the bottom of a valley. For any given state the
agent may choose to accelerate to the left, right or cease any acceleration.


Acrobot-v1
----------

`gym Acrobot-v1 source code
<https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py>`_

Acrobot is a 2-link pendulum with only the second joint actuated. Initially,
both links point downwards. The goal is to swing the end-effector at a height
at least the length of one link above the base. Both links can swing freely and
can pass by each other, i.e., they don't collide when they have the same angle.
