MetaWorld
=========

EnvPool provides native C++ implementations for the MetaWorld v3 Sawyer
manipulation benchmark. The implementation is pinned to
``Farama-Foundation/Metaworld`` tag ``v3.0.0`` and exposes one EnvPool task ID
for each official ``ALL_V3_ENVIRONMENTS`` entry.

The public task IDs use the ``MetaWorld/`` namespace and CamelCase v3 task
names, for example ``MetaWorld/Reach-v3``.

Observation and Action Spaces
-----------------------------

All MetaWorld v3 tasks share the same public space contract:

* Observation space: ``Box(-inf, inf, shape=(39,), dtype=float64)``. The vector
  contains the current 18-dimensional Sawyer observation, the previous
  18-dimensional observation, and the 3-dimensional goal vector.
* Action space: ``Box(-1, 1, shape=(4,), dtype=float32)``. The first three
  components control hand displacement and the last component controls the
  gripper.
* ``frame_skip``: 5.
* ``max_episode_steps``: 500.

Render
------

Reset render frames for all official v3 tasks are shown below. Each task panel
places the native EnvPool render on the left and the official MetaWorld v3.0.0
reference render on the right.

.. image:: ../_static/render_samples/metaworld_official_compare.png
    :width: 900px
    :align: center


Registered Task IDs
-------------------

.. code-block:: text

    MetaWorld/Assembly-v3
    MetaWorld/Basketball-v3
    MetaWorld/BinPicking-v3
    MetaWorld/BoxClose-v3
    MetaWorld/ButtonPressTopdown-v3
    MetaWorld/ButtonPressTopdownWall-v3
    MetaWorld/ButtonPress-v3
    MetaWorld/ButtonPressWall-v3
    MetaWorld/CoffeeButton-v3
    MetaWorld/CoffeePull-v3
    MetaWorld/CoffeePush-v3
    MetaWorld/DialTurn-v3
    MetaWorld/Disassemble-v3
    MetaWorld/DoorClose-v3
    MetaWorld/DoorLock-v3
    MetaWorld/DoorOpen-v3
    MetaWorld/DoorUnlock-v3
    MetaWorld/HandInsert-v3
    MetaWorld/DrawerClose-v3
    MetaWorld/DrawerOpen-v3
    MetaWorld/FaucetOpen-v3
    MetaWorld/FaucetClose-v3
    MetaWorld/Hammer-v3
    MetaWorld/HandlePressSide-v3
    MetaWorld/HandlePress-v3
    MetaWorld/HandlePullSide-v3
    MetaWorld/HandlePull-v3
    MetaWorld/LeverPull-v3
    MetaWorld/PickPlaceWall-v3
    MetaWorld/PickOutOfHole-v3
    MetaWorld/PickPlace-v3
    MetaWorld/PlateSlide-v3
    MetaWorld/PlateSlideSide-v3
    MetaWorld/PlateSlideBack-v3
    MetaWorld/PlateSlideBackSide-v3
    MetaWorld/PegInsertSide-v3
    MetaWorld/PegUnplugSide-v3
    MetaWorld/Soccer-v3
    MetaWorld/StickPush-v3
    MetaWorld/StickPull-v3
    MetaWorld/Push-v3
    MetaWorld/PushWall-v3
    MetaWorld/PushBack-v3
    MetaWorld/Reach-v3
    MetaWorld/ReachWall-v3
    MetaWorld/ShelfPlace-v3
    MetaWorld/SweepInto-v3
    MetaWorld/Sweep-v3
    MetaWorld/WindowOpen-v3
    MetaWorld/WindowClose-v3


Validation
----------

The native implementation is checked against the official MetaWorld v3.0.0
Python oracle. The alignment test reset-syncs MuJoCo state once, then drives
both implementations with the same 128-step external action sequence and
compares observations, rewards, termination flags, truncation flags, and exposed
info fields. Separate tests cover registry completeness, deterministic rollouts,
and ``rgb_array`` rendering for every registered task.
