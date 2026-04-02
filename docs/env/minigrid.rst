Minigrid
========

We use ``minigrid==3.0.0`` as the codebase.
See https://github.com/Farama-Foundation/Minigrid/tree/v3.0.0

EnvPool supports all 75 non-BabyAI ``MiniGrid-*`` environments and all 96
``BabyAI-*`` environments registered by upstream ``minigrid==3.0.0``.


Render Compare
--------------

Representative first-frame compares for the supported MiniGrid tasks. In each
panel, EnvPool is on the left and upstream ``minigrid`` is on the right.

.. image:: ../_static/render_samples/minigrid_official_compare.png
    :width: 900px
    :align: center


Options
-------

* ``task_id (str)``: see the available tasks below;
* ``num_envs (int)``: how many environments you would like to create;
* ``batch_size (int)``: the expected batch size for return result, default to
  ``num_envs``;
* ``num_threads (int)``: the maximum thread number for executing the actual
  ``env.step``, default to ``batch_size``;
* ``seed (int | Sequence[int])``: the environment seed. When a sequence is
  provided, it must contain exactly one seed per environment. Default to
  ``42``;
* ``max_episode_steps (int)``: the maximum number of steps for one episode.
  The default value depends on ``task_id`` and follows the upstream MiniGrid
  registration.


Observation Space
-----------------

Each MiniGrid observation contains:

* ``obs["image"]``: a ``(agent_view_size, agent_view_size, 3)`` uint8 tensor
  using the standard MiniGrid object/color/state encoding;
* ``obs["direction"]``: the agent direction in ``[0, 3]``;
* ``obs["mission"]``: a fixed-size uint8 byte buffer with length 96 for
  ``MiniGrid-*`` tasks and 512 for ``BabyAI-*`` tasks;
* ``info["agent_pos"]``: the agent position in the full grid;
* ``info["mission_id"]``: a stable integer ID when the mission comes from a
  finite canonical set, otherwise ``-1``.

Use ``envpool.minigrid.decode_mission(...)`` to decode the mission buffer back
to a Python string:

.. code-block:: python

   import envpool
   from envpool.minigrid import decode_mission

   env = envpool.make_gymnasium("MiniGrid-DoorKey-8x8-v0", num_envs=1)
   obs, info = env.reset()
   mission = decode_mission(obs["mission"][0])


Action Space
------------

Most tasks expose the standard MiniGrid discrete action space with values in
``[0, 6]``:

* ``0``: turn left
* ``1``: turn right
* ``2``: move forward
* ``3``: pick up an object
* ``4``: drop an object
* ``5``: toggle / interact
* ``6``: done

``MiniGrid-Dynamic-Obstacles-*`` follows upstream and only uses the movement
subset ``[0, 2]``.


Available Tasks
---------------

All upstream ``BabyAI-*`` task IDs from ``minigrid==3.0.0`` are available in
addition to the ``MiniGrid-*`` task IDs listed below.

Empty
~~~~~

* ``MiniGrid-Empty-5x5-v0``
* ``MiniGrid-Empty-Random-5x5-v0``
* ``MiniGrid-Empty-6x6-v0``
* ``MiniGrid-Empty-Random-6x6-v0``
* ``MiniGrid-Empty-8x8-v0``
* ``MiniGrid-Empty-16x16-v0``

DoorKey
~~~~~~~

* ``MiniGrid-DoorKey-5x5-v0``
* ``MiniGrid-DoorKey-6x6-v0``
* ``MiniGrid-DoorKey-8x8-v0``
* ``MiniGrid-DoorKey-16x16-v0``

DistShift
~~~~~~~~~

* ``MiniGrid-DistShift1-v0``
* ``MiniGrid-DistShift2-v0``

Crossing
~~~~~~~~

* ``MiniGrid-LavaCrossingS9N1-v0``
* ``MiniGrid-LavaCrossingS9N2-v0``
* ``MiniGrid-LavaCrossingS9N3-v0``
* ``MiniGrid-LavaCrossingS11N5-v0``
* ``MiniGrid-SimpleCrossingS9N1-v0``
* ``MiniGrid-SimpleCrossingS9N2-v0``
* ``MiniGrid-SimpleCrossingS9N3-v0``
* ``MiniGrid-SimpleCrossingS11N5-v0``

LavaGap
~~~~~~~

* ``MiniGrid-LavaGapS5-v0``
* ``MiniGrid-LavaGapS6-v0``
* ``MiniGrid-LavaGapS7-v0``

Dynamic Obstacles
~~~~~~~~~~~~~~~~~

* ``MiniGrid-Dynamic-Obstacles-5x5-v0``
* ``MiniGrid-Dynamic-Obstacles-Random-5x5-v0``
* ``MiniGrid-Dynamic-Obstacles-6x6-v0``
* ``MiniGrid-Dynamic-Obstacles-Random-6x6-v0``
* ``MiniGrid-Dynamic-Obstacles-8x8-v0``
* ``MiniGrid-Dynamic-Obstacles-16x16-v0``

Fetch
~~~~~

* ``MiniGrid-Fetch-5x5-N2-v0``
* ``MiniGrid-Fetch-6x6-N2-v0``
* ``MiniGrid-Fetch-8x8-N3-v0``

FourRooms
~~~~~~~~~

* ``MiniGrid-FourRooms-v0``

GoToDoor
~~~~~~~~

* ``MiniGrid-GoToDoor-5x5-v0``
* ``MiniGrid-GoToDoor-6x6-v0``
* ``MiniGrid-GoToDoor-8x8-v0``

GoToObject
~~~~~~~~~~

* ``MiniGrid-GoToObject-6x6-N2-v0``
* ``MiniGrid-GoToObject-8x8-N2-v0``

KeyCorridor
~~~~~~~~~~~

* ``MiniGrid-KeyCorridorS3R1-v0``
* ``MiniGrid-KeyCorridorS3R2-v0``
* ``MiniGrid-KeyCorridorS3R3-v0``
* ``MiniGrid-KeyCorridorS4R3-v0``
* ``MiniGrid-KeyCorridorS5R3-v0``
* ``MiniGrid-KeyCorridorS6R3-v0``

LockedRoom
~~~~~~~~~~

* ``MiniGrid-LockedRoom-v0``

Memory
~~~~~~

* ``MiniGrid-MemoryS17Random-v0``
* ``MiniGrid-MemoryS13Random-v0``
* ``MiniGrid-MemoryS13-v0``
* ``MiniGrid-MemoryS11-v0``
* ``MiniGrid-MemoryS9-v0``
* ``MiniGrid-MemoryS7-v0``

MultiRoom
~~~~~~~~~

* ``MiniGrid-MultiRoom-N2-S4-v0``
* ``MiniGrid-MultiRoom-N4-S5-v0``
* ``MiniGrid-MultiRoom-N6-v0``

ObstructedMaze
~~~~~~~~~~~~~~

* ``MiniGrid-ObstructedMaze-1Dl-v0``
* ``MiniGrid-ObstructedMaze-1Dlh-v0``
* ``MiniGrid-ObstructedMaze-1Dlhb-v0``
* ``MiniGrid-ObstructedMaze-2Dl-v0``
* ``MiniGrid-ObstructedMaze-2Dlh-v0``
* ``MiniGrid-ObstructedMaze-2Dlhb-v0``
* ``MiniGrid-ObstructedMaze-1Q-v0``
* ``MiniGrid-ObstructedMaze-2Q-v0``
* ``MiniGrid-ObstructedMaze-Full-v0``
* ``MiniGrid-ObstructedMaze-2Dlhb-v1``
* ``MiniGrid-ObstructedMaze-1Q-v1``
* ``MiniGrid-ObstructedMaze-2Q-v1``
* ``MiniGrid-ObstructedMaze-Full-v1``

Playground
~~~~~~~~~~

* ``MiniGrid-Playground-v0``

PutNear
~~~~~~~

* ``MiniGrid-PutNear-6x6-N2-v0``
* ``MiniGrid-PutNear-8x8-N3-v0``

RedBlueDoors
~~~~~~~~~~~~

* ``MiniGrid-RedBlueDoors-6x6-v0``
* ``MiniGrid-RedBlueDoors-8x8-v0``

Unlock
~~~~~~

* ``MiniGrid-Unlock-v0``

UnlockPickup
~~~~~~~~~~~~

* ``MiniGrid-UnlockPickup-v0``

BlockedUnlockPickup
~~~~~~~~~~~~~~~~~~~

* ``MiniGrid-BlockedUnlockPickup-v0``


Validation
----------

All registered MiniGrid task IDs are covered by:

* ``//envpool/minigrid:minigrid_align_test`` for upstream behavioral
  alignment. ``Dynamic Obstacles`` is aligned by transition replay rather than
  by sharing the exact same RNG bitstream, because upstream NumPy uses
  ``PCG64`` while EnvPool uses C++ ``mt19937``;
* ``//envpool/minigrid:minigrid_deterministic_test`` for same-seed
  determinism;
* ``//envpool:make_test`` for top-level construction coverage through the
  public ``envpool.make_*`` entry points.
