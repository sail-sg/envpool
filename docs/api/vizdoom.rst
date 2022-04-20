ViZDoom
=======

We use ``vizdoom==1.1.13`` as the codebase. See
https://github.com/mwydmuch/ViZDoom/tree/1.1.13


Env Wrappers
------------

Currently it includes these wrappers: ``frame-skip`` / ``episodic-life`` /
``action-repeat`` / ``image-resize`` / ``reward-config``.


Options
-------

* ``task_id (str)``: see available tasks below;
* ``num_envs (int)``: how many environments you would like to create;
* ``batch_size (int)``: the expected batch size for return result, default to
  ``num_envs``;
* ``num_threads (int)``: the maximum thread number for executing the actual
  ``env.step``, default to ``batch_size``;
* ``seed (int)``: the environment seed, default to ``42``;
* ``max_episode_steps (int)``: the maximum number of steps for one episode,
  default to ``525``;
* ``img_height (int)``: the desired observation image height, default to
  ``84``;
* ``img_width (int)``: the desired observation image width, default to ``84``;
* ``stack_num (int)``: the number of frames to stack for a single observation,
  default to ``4``;
* ``frame_skip (int)``: the number of frames to execute one repeated action,
  only the last frame would be kept, default to ``4``;
* ``use_inter_area_resize (bool)``: whether to use ``cv::INTER_AREA`` for
  image resize, default to ``True``;
* ``episodic_life (bool)``: make end-of-life == end-of-episode, but only reset
  on true game over. It helps the value estimation. Default to ``False``;
* ``use_combined_action (bool)``: whether to use a discrete action space as
  action input (for doom game engine), see :ref:`vizdoom_action_space`,
  default to ``False`` (use raw action space);
* ``force_speed (bool)``: if ``SPEED`` button is available, press it in every
  frame. Only available when using combined action. Default to ``False``;
* ``lmp_save_dir (str)``: the directory to save ``.lmp`` files for recording
  and replay (see tests/vizdoom/replay.py), default to ``""`` (no lmp saving);
* ``vzd_path (str)``: the vizdoom binary path, default to
  ``vizdoom/bin/vizdoom``;
* ``cfg_path (str)``: the ``.cfg`` file path, used in customized env setup,
  default to ``""``;
* ``wad_path (str)``: the ``.wad`` file path, used in customized env setup,
  default to ``""``;
* ``iwad_path (str)``: the rendering resource package path, default to
  ``vizdoom/bin/freedoom2``;
* ``map_id (str)``: the vizdoom map id, see `setDoomMap
  <https://github.com/mwydmuch/ViZDoom/blob/master/doc/DoomGame.md#setDoomMap>`_,
  available options are ``"map01", "map02", ...``, default to ``"map01"``;
* ``game_args (str)``: the args string for vizdoom game, see `addGameArgs
  <https://github.com/mwydmuch/ViZDoom/blob/master/doc/DoomGame.md#addGameArgs>`_,
  default to ``""``;
* ``reward_config (Dict[str, Tuple[float, float]])``: how to calculate the
  reward (see below), default to ``{"FRAGCOUNT": [1, -1.5], "KILLCOUNT": [1, 0],
  "DEATHCOUNT": [-0.75, 0.75], "HITCOUNT": [0.01, -0.01], "DAMAGECOUNT":
  [0.003, -0.003], "HEALTH": [0.005, -0.003], "ARMOR": [0.005, -0.001], "AMMO2":
  [0.0002, -0.0001]}``;

The original vizdoom env calculates reward with only living reward and death
penalty. Our preliminary result shows these two reward have negative effect on
agent training. Instead, other reward related to some game variable is very
useful. You can pass various reward config into vizdoom env. Each item in this
dictionary has format of ``NAME: [pos_reward, neg_reward]``. If we take
``HEALTH: [pos_health, neg_health]`` as an example:

::

    delta = current[HEALTH] - last[HEALTH]
    if delta >= 0:
        reward += delta * pos_health
    else:
        reward -= delta * neg_health

where ``last[*]`` is the corresponding value at the last timestep.

* ``weapon_duration`` see below, default to ``5``;
* ``selected_weapon_reward_config (Dict[int, float])``: the available keys are
  ``2 / 3 / ... / 7``, it means if the agent holds ``i``th weapon for at least
  ``weapon_duration`` timestep, the reward will be added by
  ``selected_weapon_reward_config[i]``;
* ``delta_button_config (Dict[str, Tuple[int, float, float]])``: see
  :ref:`vizdoom_action_space`.


Customized VizDoom Env
----------------------

Use ``VizdoomCustom-v1`` with ``cfg_path`` and ``wad_path``:
::

   env = envpool.make(
     "VizdoomCustom-v1",
     cfg_path="xxx.cfg",
     wad_path="xxx.wad",
     ...
    )


Observation Space
-----------------

The observation channel number is defined in configuration file (e.g.,
``GRAY8`` or ``CRCGCB``).

The game variables defined in configuration file are in observation (dm) / are
in info (gym). Each variable tags a key. Currently it includes:
``"AMMO2", "AMMO3", "AMMO4", "AMMO5", "AMMO6", "AMMO7", "ARMOR",
"DAMAGECOUNT", "DEATHCOUNT", "FRAGCOUNT", "HEALTH", "HITCOUNT",
"KILLCOUNT", "SELECTED_WEAPON", "SELECTED_WEAPON_AMMO", "USER2"``

.. _vizdoom_action_space:

Action Space
------------

If ``use_combined_action`` is set to ``False``, it only accepts the original
action input (which is a list of float); otherwise:

All of the buttons are in discrete space, including delta button. In EnvPool
we directly generate the combo action with the following rule:

1. Each time the agent can only select at most one weapon

   * can only select at most one of ``SELECT_WEAPON0`` ... ``SELECT_WEAPON9``
     buttons

2. Some buttons are pair-wised, they cannot be selected together (``FF, TF, FT``)

   * ``MOVE_LEFT`` and ``MOVE_RIGHT``, ``MOVE_FORWARD`` and ``MOVE_BACKWARD``,
     ``TURN_LEFT`` and ``TURN_RIGHT``, ``LOOK_UP`` and ``LOOK_DOWN``,
     ``MOVE_UP`` and ``MOVE_DOWN``, ``SELECT_PREV_WEAPON`` and
     ``SELECT_NEXT_WEAPON``, ``SELECT_PREV_ITEM`` and ``SELECT_NEXT_ITEM``

3. Other non-delta buttons have two choices: ``F`` or ``T``

4. For delta buttons, the given ``delta_button_config`` specifies how it builds
   the action set (with format ``[num, min, max]``). For example, if we pass
   ``delta_button_config={"TURN_LEFT_RIGHT_DELTA": [4, -2.0, 1.0]}``, it will
   build ``TURN_LEFT_RIGHT_DELTA = [-2.0, -1.0, 0.0, 1.0]`` 4 discrete choices.

For example, if we have ``MOVE_FORWARD``, ``TURN_LEFT`` and ``TURN_RIGHT``
three buttons (which is exactly health-gathering setting), we have 2x3=6
discrete actions according the above rule.


Available Tasks
---------------

* ``Basic-v1``
* ``Cig-v1``
* ``D1Basic-v1``
* ``D2Navigation-v1``
* ``D3Battle-v1``
* ``D4Battle2-v1``
* ``DeadlyCorridor-v1``
* ``Deathmatch-v1``
* ``DefendTheCenter-v1``
* ``DefendTheLine-v1``
* ``HealthGathering-v1``
* ``HealthGatheringSupreme-v1``
* ``MultiDuel-v1``
* ``MyWayHome-v1``
* ``PredictPosition-v1``
* ``RocketBasic-v1``
* ``SimplerBasic-v1``
* ``TakeCover-v1``
* ``VizdoomCustom-v1``
