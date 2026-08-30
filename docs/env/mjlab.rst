MJLab
=====

EnvPool implements the twelve built-in training tasks from
`MJLab 1.6.0 <https://github.com/mujocolab/mjlab/tree/b517e0c489139e7fcee95702cfb2b01931264985>`_
in C++. Each upstream ID has a versioned EnvPool ID and an alias without the suffix:
``Mjlab-Cartpole-Balance-v0`` and ``Mjlab-Cartpole-Balance``, for example.

The physics and observation cameras use ahead-of-time CPU kernels from
MuJoCo-Warp 3.11.0 and Warp 1.14.0. This preserves the upstream float32
simulation path. Python MJLab, Torch, a GPU, and a runtime kernel compiler are
not required to use these environments. The ordinary MuJoCo renderer provides
the separate public ``render()`` images.

Models, textures, and serialized CPU graphs are installed by
``envpool-assets-mjlab``. They are shared between tasks without data loss and kept out
of the core EnvPool wheel. Motion recordings are user inputs and are not
bundled in that asset package.

Task Coverage
-------------

Observations are dictionaries with ``actor`` and ``critic`` entries, plus
``camera`` for the three visual manipulation tasks. All entries and actions
use ``float32``. Action spaces are unbounded, matching the upstream presets;
the native action manager applies each preset's scaling and actuator limits.
Shapes below exclude the leading environment dimension.

Every ID in this table also accepts the ``-v0`` suffix.

.. list-table::
   :header-rows: 1
   :widths: 48 9 9 9 17 8

   * - Task ID
     - Action
     - Actor
     - Critic
     - Camera
     - Steps
   * - ``Mjlab-Cartpole-Balance``
     - 1
     - 5
     - 5
     - none
     - 1000
   * - ``Mjlab-Cartpole-Swingup``
     - 1
     - 5
     - 5
     - none
     - 1000
   * - ``Mjlab-Lift-Cube-Yam``
     - 7
     - 29
     - 29
     - none
     - 1000
   * - ``Mjlab-Lift-Cube-Yam-Depth``
     - 7
     - 26
     - 29
     - ``(1, 32, 32)``
     - 1000
   * - ``Mjlab-Lift-Cube-Yam-Rgb``
     - 7
     - 26
     - 29
     - ``(3, 32, 32)``
     - 1000
   * - ``Mjlab-Multi-Cube-Seg-Yam``
     - 7
     - 26
     - 26
     - ``(2, 32, 32)``
     - 1000
   * - ``Mjlab-Tracking-Flat-Unitree-G1``
     - 29
     - 160
     - 286
     - none
     - 500
   * - ``Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation``
     - 29
     - 154
     - 286
     - none
     - 500
   * - ``Mjlab-Velocity-Flat-Unitree-G1``
     - 29
     - 99
     - 111
     - none
     - 1000
   * - ``Mjlab-Velocity-Flat-Unitree-Go1``
     - 12
     - 48
     - 72
     - none
     - 1000
   * - ``Mjlab-Velocity-Rough-Unitree-G1``
     - 29
     - 286
     - 298
     - none
     - 1000
   * - ``Mjlab-Velocity-Rough-Unitree-Go1``
     - 12
     - 235
     - 259
     - none
     - 1000

The rough-terrain presets include flat ground, ascending and descending
stairs, ascending and descending slopes, random rough ground, and waves.
Terrain columns are distributed across the pool according to the upstream
proportions; a pool with at least seven environments includes all columns.
Training noise, domain randomization, command resampling, and curricula remain
enabled. The upstream play configurations and external task plugins are not
additional registered tasks in this family.

Rendered Examples
-----------------

Each row compares EnvPool on the left with official MJLab on the right after
identical deterministic actions. The image generator first verifies the
complete episode and its rendered frames. Rough-terrain examples use the wave
column; tracking examples use the generated test motion, without a trained policy.

.. figure:: /_static/render_samples/mjlab-cartpole.png
   :width: 960
   :alt: Native and official Cartpole Balance and Swingup renders.

   Cartpole tasks.

.. figure:: /_static/render_samples/mjlab-manipulation.png
   :width: 960
   :alt: Native and official renders of all four Yam manipulation tasks.

   Yam manipulation tasks. These show the public scene renderer; observation
   cameras are separate and are checked by the alignment tests as well.

.. figure:: /_static/render_samples/mjlab-tracking.png
   :width: 960
   :alt: Native and official renders of both G1 motion tracking tasks.

   G1 motion tracking tasks.

.. figure:: /_static/render_samples/mjlab-velocity.png
   :width: 960
   :alt: Native and official G1 and Go1 renders on flat and wave terrain.

   G1 and Go1 velocity tasks.

Usage
-----

.. code-block:: python

   import envpool
   import numpy as np

   env = envpool.make_gymnasium(
       "Mjlab-Velocity-Flat-Unitree-Go1-v0",
       num_envs=4,
       num_threads=2,
       seed=42,
       render_mode="rgb_array",
       render_width=320,
       render_height=240,
   )
   try:
       obs, info = env.reset()
       action = np.zeros((4, *env.action_space.shape), dtype=np.float32)
       obs, reward, terminated, truncated, info = env.step(action)
       frames = env.render(env_ids=[0, 2])  # uint8, (2, 240, 320, 3)
   finally:
       env.close()

``make_dm`` exposes the same dictionary through ``timestep.observation.obs``.
An episode termination sets the discount to zero; a time-limit truncation
keeps it at one. As in other EnvPool families, stepping a completed slot resets
it before the next episode. ``max_episode_steps`` may shorten the preset's
episode limit. ``info`` contains EnvPool slot identifiers and ``elapsed_step``;
MJLab's training logger and Weights & Biases integration are not part of this API.

Each slot has its own RNG and episode state. Curriculum steps and motion
failure statistics are maintained per slot, so asynchronous scheduling does
not couple one slot's reset distribution to another slot's progress. This
corresponds to independent upstream single-environment instances, rather than
the shared adaptive histogram in one upstream training batch.
Terrain-column allocation still uses the pool size as described above.

Motion Tracking Input
---------------------

Both tracking tasks require ``motion_file="/path/to/motion.npz"``. Upstream
does not supply a default recording, and EnvPool does not download one or
contact a motion registry. Use the pinned MJLab motion format, with these
arrays in the robot's joint and body ordering:

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Array
     - Shape
     - Meaning
   * - ``joint_pos``, ``joint_vel``
     - ``(T, 29)``
     - G1 joint positions and velocities
   * - ``body_pos_w``
     - ``(T, 30, 3)``
     - Body positions in world coordinates
   * - ``body_quat_w``
     - ``(T, 30, 4)``
     - Body orientations in ``wxyz`` order
   * - ``body_lin_vel_w``, ``body_ang_vel_w``
     - ``(T, 30, 3)``
     - Body linear and angular velocities in world coordinates

Use 50 Hz samples: the reference advances one frame per 0.02-second control
step, as in the pinned upstream task. All six arrays must have the same
positive frame count and finite numeric values. Stored and deflated NPZ files,
NumPy ``C`` and ``Fortran`` array orders, both byte orders, and float16/32/64 values are
accepted and converted to float32. Object arrays are rejected. The native
reader checks archive sizes, array shapes, and data integrity before simulation.

The test fixture is generated locally using the pinned upstream CSV conversion
code. It exercises moving references and short-recording wraparound, but is
not a motion dataset or a default training benchmark.

Building from source
--------------------

The official SDK is needed only to generate C++ sources and model data, not to
run EnvPool. Its ARM Linux wheel requires ``glibc`` 2.34 or newer. Release builds
therefore generate portable inputs on Ubuntu, then compile them inside
``manylinux`` 2.28. This keeps the existing wheel compatibility floor.

For a source build on an older Linux system, prepare the usual build
dependencies on a supported host, then run:

.. code-block:: bash

   make bazel-pip-requirement-dev
   bazel build --config=test //third_party/mjlab:codegen_bundle

Copy ``bazel-bin/third_party/mjlab/codegen/codegen_input.tar.gz`` into
``third_party/mjlab/`` in the same source revision. Bazel validates the generator
recipe and every archived file before using it. Remove that input to regenerate
locally. Neither the archive nor the SDK is included in a released wheel.

Validation
----------

The CPU oracle uses unmodified MJLab 1.6.0, Torch 2.9.0, MuJoCo-Warp 3.11.0, and
Warp 1.14.0. It synchronizes native state once immediately after reset, then
compares complete episodes driven by identical external actions. Tests cover
every task, observations including camera pixels, rewards, termination,
truncation, physics state, and public rendering at multiple steps.
The oracle uses the upstream ``auto_reset=False`` option to retain terminal
observations; EnvPool resets a completed slot on its next step.

Reward reductions use standard C++ math; the terrain-plane fit uses MuJoCo's
3-by-3 eigensolver. Only arithmetic and trigonometry that set physical state
retain the pinned oracle's rounding, since tiny changes there can grow into
different contact trajectories. Sine and cosine use statically linked MKL on
x64 and SLEEF on ARM64; no additional math runtime or LAPACK solver is needed.
Oracle comparisons allow float32 rounding differences (relative tolerance
``1e-5``, absolute tolerance ``1e-6``), while discrete outputs and native
same-seed replays remain exact. The tests still compare every step through
termination; they do not substitute final scores for behavioral alignment.

RGB rendering shares the existing MuJoCo bootstrap. On macOS, identical scenes
can produce sparse CGL/Metal color differences. The shared image check allows
at most five intensity levels in any color channel and a mean absolute error
of ``0.01`` per frame, on the 0-to-255 scale. Other platforms retain exact RGB
comparisons. No extra per-frame draw or task-specific pixel limit is required.

Independent tests observe native resets without oracle synchronization. They
check different seeds, consecutive resets, parallel slots, and replay of the
whole reset sequence. Independently randomized goals, poses, model properties,
and terrain are checked separately so observation noise cannot conceal a
frozen task component. This preserves the regression boundary from
`issue 432 <https://github.com/sail-sg/envpool/issues/432>`_.
