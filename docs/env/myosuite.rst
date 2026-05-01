MyoSuite
========

EnvPool's MyoSuite integration uses ``myosuite==2.11.6`` pinned at commit
``05cb84678373f91271004f99602ebbf01e57d1a1`` with ``mujoco==3.6.0``.
The runtime implementation is native C++; the official Python package is used
only by oracle tests and doc-generation tooling.

The generated upstream registry and task metadata live under
``third_party/myosuite/``. Runtime C++ consumes those generated assets instead
of keeping a handwritten task list in ``envpool/mujoco/myosuite/``.


Env IDs
-------

EnvPool registers all 398 official MyoSuite task IDs from the pinned oracle.
Every official ID also has an EnvPool alias of the form
``MyoSuite/<official-id>``, for example:

::

  envpool.make_gymnasium("myoFingerReachFixed-v0")
  envpool.make_gymnasium("MyoSuite/myoFingerReachFixed-v0")

The covered surface includes MyoBase reach, pose, key-turn, object-hold,
pen-twirl, reorient, walk, and terrain tasks; MyoChallenge tasks; MyoDM track
tasks; and the corresponding normal, sarcopenia, fatigue, and
reafferentation variants exposed by upstream.

Nine upstream IDs are still registered in EnvPool but are excluded from
official-oracle alignment tests because the pinned official package cannot
instantiate them under the MuJoCo 3.6 oracle environment:

::

  myoChallengeBimanual-v0
  myoSarcChallengeBimanual-v0
  myoFatiChallengeBimanual-v0
  myoChallengeSoccerP1-v0
  myoChallengeSoccerP2-v0
  myoSarcChallengeSoccerP1-v0
  myoSarcChallengeSoccerP2-v0
  myoFatiChallengeSoccerP1-v0
  myoFatiChallengeSoccerP2-v0


Render Compare
--------------

Reset and first-three-step render comparisons for every pinned official task
that the upstream oracle can instantiate. For each step pair, EnvPool is on the
left and the pinned MyoSuite renderer is on the right. The images are generated
by ``third_party/myosuite/generate_render_sample.py`` from the pinned official
oracle and the same action sequence used by the render test. The nine upstream
Bimanual/Soccer IDs listed above remain registered but are omitted from these
official render sheets for the same oracle instantiation failure.

.. image:: ../_static/render_samples/myosuite_myobase_official_compare.png
    :width: 900px
    :align: center

.. image:: ../_static/render_samples/myosuite_myochallenge_official_compare.png
    :width: 900px
    :align: center

.. image:: ../_static/render_samples/myosuite_myodm_official_compare.png
    :width: 900px
    :align: center
