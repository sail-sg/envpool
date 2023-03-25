Procgen
=======

We use ``procgen==0.10.7`` as the codebase.
See https://github.com/openai/procgen/tree/0.10.7


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
  each procgen game has different timeout value;
* ``channel_first (bool)``: whether to transpose the observation image to
  ``(3, 64, 64)``, default to ``True``;
* ``env_name (str)``: one of 16 procgen env name;
* ``num_levels (int)``: default to ``0``;
* ``start_level (int)``: default to ``0``;
* ``use_sequential_levels (bool)``: default to ``False``;
* ``center_agent (bool)``: default to ``True``;
* ``use_backgrounds (bool)``: default to ``True``;
* ``use_monochrome_assets (bool)``: default to ``False``;
* ``restrict_themes (bool)``: default to ``False``;
* ``use_generated_assets (bool)``: default to ``False``;
* ``paint_vel_info (bool)``: default to ``False``;
* ``use_easy_jump (bool)``: default to ``False``;
* ``distribution_mode (int)``: one of ``(0, 1, 2, 10)``; ``0`` stands for easy
  mode, ``1`` stands for hard mode, ``2`` stands for extreme mode, ``10``
  stands for memory mode. The default value is determined by ``task_id``.

Note: arguments after ``env_name`` are provided by procgen environment itself.
We keep the default value as-is. We haven't tested the setting of
``use_sequential_levels == True``, and have no promise it is aligned with the
original version of procgen (PRs for fixing this issue are highly welcome).


Observation Space
-----------------

The observation image shape is ``(3, 64, 64)`` when ``channel_first`` is
``True`` (default), ``(64, 64, 3)`` when ``channel_first`` is ``False``.


Action Space
------------

15 action buttons in total, ranging from 0 to 14.


Available Tasks
---------------

* ``BigfishEasy-v0``
* ``BigfishHard-v0``
* ``BossfightEasy-v0``
* ``BossfightHard-v0``
* ``CaveflyerEasy-v0``
* ``CaveflyerHard-v0``
* ``CaveflyerMemory-v0``
* ``ChaserEasy-v0``
* ``ChaserHard-v0``
* ``ChaserExtreme-v0``
* ``ClimberEasy-v0``
* ``ClimberHard-v0``
* ``CoinrunEasy-v0``
* ``CoinrunHard-v0``
* ``DodgeballEasy-v0``
* ``DodgeballHard-v0``
* ``DodgeballExtreme-v0``
* ``DodgeballMemory-v0``
* ``FruitbotEasy-v0``
* ``FruitbotHard-v0``
* ``HeistEasy-v0``
* ``HeistHard-v0``
* ``HeistMemory-v0``
* ``JumperEasy-v0``
* ``JumperHard-v0``
* ``JumperMemory-v0``
* ``LeaperEasy-v0``
* ``LeaperHard-v0``
* ``LeaperExtreme-v0``
* ``MazeEasy-v0``
* ``MazeHard-v0``
* ``MazeMemory-v0``
* ``MinerEasy-v0``
* ``MinerHard-v0``
* ``MinerMemory-v0``
* ``NinjaEasy-v0``
* ``NinjaHard-v0``
* ``PlunderEasy-v0``
* ``PlunderHard-v0``
* ``StarpilotEasy-v0``
* ``StarpilotHard-v0``
* ``StarpilotExtreme-v0``
