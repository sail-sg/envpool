Craftax
========

`Craftax <https://github.com/MichaelTMatthews/Craftax>`_ is a survival and
crafting environment. Craftax Classic contains the original single-world
survival game. Full Craftax adds nine floors, dungeons, equipment, elemental
combat, spells, potions, and a final boss.

EnvPool implements both games and their renderers in C++. The reference is
Craftax v1.6.1, commit ``c3c2e0d038c4e641f9481320c158f457f30c28f3``.
Craftax, JAX, and Flax are not runtime dependencies. The extension embeds only
PNG textures referenced by the official renderer; no Python game code or
training assets are loaded at runtime. The upstream MIT license is included
in the wheel.

Environments
------------

.. list-table:: Primary environment names
   :header-rows: 1
   :widths: 45 12 25 18

   * - Name
     - Actions
     - Observation shape
     - Time limit
   * - ``Craftax-Classic-Symbolic-v1``
     - 17
     - ``(1345,)``
     - 10,000
   * - ``Craftax-Classic-Pixels-v1``
     - 17
     - ``(63, 63, 3)``
     - 10,000
   * - ``Craftax-Symbolic-v1``
     - 43
     - ``(8268,)``
     - 100,000
   * - ``Craftax-Pixels-v1``
     - 43
     - ``(130, 110, 3)``
     - 100,000

All observations are ``float32``. Pixel observations use the official
normalization to ``[0, 1]`` and include the inventory display. Symbolic
observations contain the local map, entities, inventory, and player status.
The declared symbolic bounds follow upstream; some equipment and status
features can exceed one during play. Only visible, active entities contribute
to the symbolic view.

Each name also has the official ``-AutoReset-v1`` variant, for example
``Craftax-Symbolic-AutoReset-v1``. Every name has a ``Craftax/`` alias:
``Craftax/Symbolic-v1`` and ``Craftax/Classic-Pixels-v1`` are examples.
The registry is generated from the pinned upstream factory.

Usage
-----

.. code-block:: python

    import envpool
    import numpy as np

    env = envpool.make_gymnasium(
        "Craftax-Symbolic-v1",
        num_envs=32,
        seed=0,
        render_mode="rgb_array",
    )
    obs, info = env.reset()
    actions = np.zeros(32, dtype=np.int32)
    obs, reward, terminated, truncated, info = env.step(actions)
    frames = env.render(env_ids=[0, 3])
    env.close()

Reset and termination
---------------------

Primary names use EnvPool's usual reset on the step after an episode ends.
That step returns the new initial observation, zero reward, and a first
``dm_env`` timestep; its submitted action is not applied.

The ``AutoReset`` names reset on the terminal step itself, as the official
Craftax wrapper does. They return the new initial observation with the old
episode's reward, terminal flag, and achievement information. The next action
is applied to the new episode. Rendering after that step shows the new state.
These names advertise ``SameStep`` in Gymnasium's autoreset metadata.

Death, Classic lava, and full-game boss victory are terminations. A time
limit without a simultaneous game termination is a truncation. The discount
is zero at either boundary, matching the official oracle, including in the
``dm_env`` API. ``info["discount"]`` retains that same value.

Achievement information uses the official ``Achievements/<name>`` keys and
reports 100 for an earned achievement only at the episode boundary. Classic
also returns the official geometric-mean ``score``. In ``dm_env`` namedtuples,
the usual EnvPool conversion replaces slashes in field names with underscores.

Configuration and reproducibility
---------------------------------

``max_episode_steps`` sets upstream ``max_timesteps``. Both games accept
``day_length``, ``always_diamond``, ``mob_despawn_distance``, entity capacities,
and ``fractal_noise_angles``. Classic uses the shared capacity names
``max_melee_mobs``, ``max_passive_mobs``, ``max_ranged_mobs``, and
``max_mob_projectiles`` for zombies, cows, skeletons, and arrows respectively.
Classic also accepts the official mob health and spawn chance parameters.
``god_mode`` and ``max_attribute`` apply to the full game.

``map_size`` defaults to ``(64, 64)`` for Classic and ``(48, 48)`` for the full
game. Maps must be square with a size divisible by 16, at least 16 for Classic
and 48 for full Craftax's eight-room dungeon generator. The number of floors
is fixed at one and nine respectively. Noise overrides are four flattened
arrays in upstream order; an empty array selects random angles for that layer.

Craftax uses EnvPool's native random-number generator. Each environment
starts its own stream from ``seed + env_id`` or its explicit ``env_seed``.
Resets and steps advance that stream. On the same EnvPool build and
platform, recreating the pool with the same configuration and seed and
replaying the same actions reproduces the rollout across thread counts.

Seeds from versions that used JAX-compatible sampling produce different
worlds and trajectories with the native generator. Keep the original EnvPool
version when replaying a trajectory from one of those versions.

Rendering
---------

``render()`` produces batched ``uint8`` RGB frames. With the default
``render_tile_size=16``, Classic frames are ``(144, 144, 3)`` and full Craftax
frames are ``(208, 176, 3)``. ``render_tile_size=64`` selects the official human
texture resolution. The standard ``render_width`` and ``render_height``
options resize the resulting frame by nearest-neighbor sampling.

The renderer includes the official sprites, inventory digits, projectiles,
light and night effects, sleep shading, and full-game floor visibility.

.. image:: /_static/render_samples/craftax_official_compare.png
   :alt: EnvPool on the left and official Craftax on the right after gameplay

EnvPool is on the left and the pinned official renderer is on the right.
The documentation generator checks the rendered scene after each displayed
action sequence, allowing one RGB level for quantization and float32 rounding.

Regenerate it with the shared documentation tool:

.. code-block:: bash

   bazel run --config=test //scripts:render_compare -- \
     --family=craftax --seed=11 --columns=1 --max-ignored-abs-diff=1 \
     --tile-width=352 --tile-height=416

Validation
----------

Oracle tests use the pinned source and a separate dependency lock, including
JAX/JAXlib 0.11.1 and Flax 0.12.9. They exercise all factory names, whole
trajectories, crafting and resource interactions, every floor's combat,
projectiles, potion and enchantment actions, plants, boss waves, and victory.
The official texture cache is a test-only build artifact and is never shipped
as a runtime dependency.

Native tests check default seed variation and replay across resets and
parallel slots. They also check the generated floors, potion assignments,
and the rates and legal outcomes of native spawns, movement, and loot.
Oracle rollouts start from the native world after checking the generated
terrain, ladders, lighting, and items. Both implementations step the same
actions independently through episode end, with random spawns and loot
covered in separate tests. Directed fixtures exercise combat and specific
actions from a known initial state, including Classic's activation of
positive-health mob slots during interaction. In the chest fixtures, the
tests compare the opening, guaranteed rewards, and transition outputs
before copying the separately checked random loot for the rest of the rollout.

The ``initial_state`` configuration accepts a diagnostic snapshot for the
first reset. Use ``debug_state=True`` to expose that encoding in
``info["state"]``. The encoding is intended for diagnostics and may change
between versions.

Scalar native math and the official JAX graphs may round daylight blends,
normalized observations, and the achievement score differently. Oracle tests
use float32 accuracy (relative tolerance ``1e-6``, absolute tolerance ``1e-7``),
including pixels normalized to [0, 1]. Integer RGB output allows one intensity
level for quantization. These criteria apply across architectures and channels.

Game state is compared by field: discrete maps, inventory, achievements, and
transitions remain exact; floating-point fields use the numerical criterion
above. Tests cover complete day/night and AutoReset trajectories. At night
the oracle bounds the scene and lighting using the endpoints of its pixel
noise distribution. Native tests check the noise distribution and exact
same-state render replay. Repeated native runs retain exact state and
observation equality.
