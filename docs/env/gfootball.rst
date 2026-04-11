Google Research Football
========================

EnvPool supports the Google Research Football scenarios listed below under
EnvPool-native IDs of the form ``gfootball/<scenario>-v1``. The runtime path is
pure C++: EnvPool builds and calls the upstream engine directly, and only uses
the upstream Python scenario files at build time to generate a static C++
scenario table. There is no Python-environment bridge in the runtime hot path.


Options
-------

* ``task_id (str)``: any task listed in `Available Tasks`_;
* ``num_envs (int)``: how many environments to create;
* ``batch_size (int)``: the expected batch size for returned results, default
  to ``num_envs``;
* ``num_threads (int)``: the maximum worker count used for ``env.step``,
  default to ``batch_size``;
* ``seed (int | Sequence[int])``: the environment seed. When a sequence is
  provided, it must contain exactly one seed per environment. Default to
  ``42``;
* ``max_episode_steps (int)``: the episode horizon. Each task defaults to the
  upstream scenario duration, but callers can override it;
* ``render_mode (str)``: ``"rgb_array"`` for batched RGB rendering or
  ``"human"`` for the OpenCV viewer;
* ``render_width (int)`` and ``render_height (int)``: output render size. The
  default engine resolution is ``1280 x 720``;
* ``physics_steps_per_frame (int)``: physics updates per policy step, default
  to ``10``.


Observation Space
-----------------

The Gymnasium wrapper exposes a ``uint8`` minimap tensor ``obs`` with shape
``(72, 96, 4)``:

* channel ``0`` marks left-team players;
* channel ``1`` marks right-team players;
* channel ``2`` marks the ball position;
* channel ``3`` marks the currently controlled left-team player.

The ``info`` dict contains:

* ``score``: ``int32[2]`` with ``[left_goals, right_goals]``;
* ``game_mode``: current upstream game-mode enum value;
* ``ball_owned_team`` and ``ball_owned_player``;
* ``steps_left`` and ``elapsed_step``;
* ``engine_seed`` and ``episode_number``.

Rewards follow upstream score-delta semantics: each step returns the change in
``left_goals - right_goals`` since the previous observation.


Action Space
------------

The action space is discrete with 19 actions matching EnvPool's default Google
Research Football action set:

* ``0``: ``idle``
* ``1``: ``left``
* ``2``: ``top_left``
* ``3``: ``top``
* ``4``: ``top_right``
* ``5``: ``right``
* ``6``: ``bottom_right``
* ``7``: ``bottom``
* ``8``: ``bottom_left``
* ``9``: ``long_pass``
* ``10``: ``high_pass``
* ``11``: ``short_pass``
* ``12``: ``shot``
* ``13``: ``sprint``
* ``14``: ``release_direction``
* ``15``: ``release_sprint``
* ``16``: ``sliding``
* ``17``: ``dribble``
* ``18``: ``release_dribble``


Available Tasks
---------------

* ``gfootball/11_vs_11_competition-v1``
* ``gfootball/11_vs_11_easy_stochastic-v1``
* ``gfootball/11_vs_11_hard_stochastic-v1``
* ``gfootball/11_vs_11_kaggle-v1``
* ``gfootball/11_vs_11_stochastic-v1``
* ``gfootball/1_vs_1_easy-v1``
* ``gfootball/5_vs_5-v1``
* ``gfootball/academy_3_vs_1_with_keeper-v1``
* ``gfootball/academy_corner-v1``
* ``gfootball/academy_counterattack_easy-v1``
* ``gfootball/academy_counterattack_hard-v1``
* ``gfootball/academy_empty_goal-v1``
* ``gfootball/academy_empty_goal_close-v1``
* ``gfootball/academy_pass_and_shoot_with_keeper-v1``
* ``gfootball/academy_run_pass_and_shoot_with_keeper-v1``
* ``gfootball/academy_run_to_score-v1``
* ``gfootball/academy_run_to_score_with_keeper-v1``
* ``gfootball/academy_single_goal_versus_lazy-v1``


Correctness Tests
-----------------

The gfootball integration is held to the same no-skip, all-platform standard as
the rest of EnvPool's core environments:

* Full registry coverage: every registered ``gfootball/*-v1`` ID is created
  through the public Gymnasium API, reset, stepped, and closed in CI;
* Bitwise rollout alignment: ``envpool/gfootball/gfootball_align_test.py``
  checks observations, rewards, ``terminated`` / ``truncated``, and the public
  ``info`` fields bitwise against an independent oracle that uses the same C++
  engine with upstream scenario semantics;
* Bitwise render alignment: ``envpool/gfootball/gfootball_render_test.py``
  checks the reset frame and multiple step frames bitwise for every registered
  task;
* Determinism: ``envpool/gfootball/gfootball_deterministic_test.py`` requires
  identical seeds and action streams to produce identical rollouts, while a
  different seed must change the rollout;
* Cross-platform coverage: the build, render, alignment, and determinism paths
  are validated in CI on Linux, macOS, and Windows rather than being skipped on
  any platform.
