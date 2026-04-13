MetaWorld
=========

EnvPool provides native C++ implementations for the MetaWorld v3 Sawyer
manipulation benchmark. The implementation is pinned to
``Farama-Foundation/Metaworld`` tag ``v3.0.0`` and exposes one EnvPool task ID
for each official ``ALL_V3_ENVIRONMENTS`` entry.

The public task IDs use the official ``Meta-World/`` namespace and the
official lower-case, hyphenated v3 task names, for example
``Meta-World/reach-v3``. The historical EnvPool spelling ``MetaWorld/`` is
also registered as an alias for each task.

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

    Meta-World/assembly-v3
    Meta-World/basketball-v3
    Meta-World/bin-picking-v3
    Meta-World/box-close-v3
    Meta-World/button-press-topdown-v3
    Meta-World/button-press-topdown-wall-v3
    Meta-World/button-press-v3
    Meta-World/button-press-wall-v3
    Meta-World/coffee-button-v3
    Meta-World/coffee-pull-v3
    Meta-World/coffee-push-v3
    Meta-World/dial-turn-v3
    Meta-World/disassemble-v3
    Meta-World/door-close-v3
    Meta-World/door-lock-v3
    Meta-World/door-open-v3
    Meta-World/door-unlock-v3
    Meta-World/hand-insert-v3
    Meta-World/drawer-close-v3
    Meta-World/drawer-open-v3
    Meta-World/faucet-open-v3
    Meta-World/faucet-close-v3
    Meta-World/hammer-v3
    Meta-World/handle-press-side-v3
    Meta-World/handle-press-v3
    Meta-World/handle-pull-side-v3
    Meta-World/handle-pull-v3
    Meta-World/lever-pull-v3
    Meta-World/pick-place-wall-v3
    Meta-World/pick-out-of-hole-v3
    Meta-World/pick-place-v3
    Meta-World/plate-slide-v3
    Meta-World/plate-slide-side-v3
    Meta-World/plate-slide-back-v3
    Meta-World/plate-slide-back-side-v3
    Meta-World/peg-insert-side-v3
    Meta-World/peg-unplug-side-v3
    Meta-World/soccer-v3
    Meta-World/stick-push-v3
    Meta-World/stick-pull-v3
    Meta-World/push-v3
    Meta-World/push-wall-v3
    Meta-World/push-back-v3
    Meta-World/reach-v3
    Meta-World/reach-wall-v3
    Meta-World/shelf-place-v3
    Meta-World/sweep-into-v3
    Meta-World/sweep-v3
    Meta-World/window-open-v3
    Meta-World/window-close-v3


Validation
----------

The native implementation is checked against the official MetaWorld v3.0.0
Python oracle. The alignment test reset-syncs MuJoCo state once, then drives
both implementations with the same external action sequence and compares
observations, rewards, termination flags, truncation flags, and exposed info
fields. Separate tests cover registry completeness, deterministic rollouts, and
``rgb_array`` rendering for every registered task.
