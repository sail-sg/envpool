Atari
=====

We use ``ale_py>=0.7.2`` as the codebase.
See https://github.com/mgbellemare/Arcade-Learning-Environment/tree/v0.7.2


Env Wrappers
------------

Currently it includes these wrappers: ``random-noops`` / ``fire-reset`` /
``episodic-life`` / ``frame-skip`` / ``action-repeat`` / ``image-resize`` /
``reward-clip``. The wrapper execution order is the same as
`OpenAI Baselines <https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py>`_.


Options
-------

* ``task_id (str)``: see available tasks below;
* ``num_envs (int)``: how many environments you would like to create;
* ``batch_size (int)``: the expected batch size for return result, default to
  ``num_envs``;
* ``num_threads (int)``: the maximum thread number for executing the actual
  ``env.step``, default to ``batch_size``;
* ``seed (int)``: the environment seed, default to ``0``;
* ``max_episode_steps (int)``: the maximum number of steps for one episode,
  default to ``25000``;
* ``img_height (int)``: the desired observation image height, default to
  ``84``;
* ``img_width (int)``: the desired observation image width, default to ``84``;
* ``stack_num (int)``: the number of frames to stack for a single observation,
  default to ``4``;
* ``frame_skip (int)``: the number of frames to execute one repeated action,
  only the last frame would be kept, default to ``4``;
* ``noop_max (int)``: the maximum number of no-op action being executed when
  calling a single ``env.reset``, default to ``30``;
* ``episodic_life (bool)``: make end-of-life == end-of-episode, but only reset
  on true game over. It helps the value estimation. Default to ``False``;
* ``zero_discount_on_life_loss (bool)``: when the agent losses a life, the
  ``discount`` in dm_env.TimeStep is set to 0. This option doesn't affect gym's
  behavior (since there is no ``discount`` field in gym's API). Default to
  ``False``;
* ``reward_clip (bool)``: whether to change the reward to ``sign(reward)``,
  default to ``False``.

Observation Space
-----------------

The observation image size should be ``(stack_num, img_height, img_width)``,
``(4, 84, 84)`` by default. For a single frame, it has been gray-scaled and
resized inside the c++ code.


Action Space
------------

Each Atari games has its own discrete action space.


Available Tasks
---------------

* ``Adventure-v5``
* ``AirRaid-v5``
* ``Alien-v5``
* ``Amidar-v5``
* ``Assault-v5``
* ``Asterix-v5``
* ``Asteroids-v5``
* ``Atlantis-v5``
* ``Atlantis2-v5``
* ``Backgammon-v5``
* ``BankHeist-v5``
* ``BasicMath-v5``
* ``BattleZone-v5``
* ``BeamRider-v5``
* ``Berzerk-v5``
* ``Blackjack-v5``
* ``Bowling-v5``
* ``Boxing-v5``
* ``Breakout-v5``
* ``Carnival-v5``
* ``Casino-v5``
* ``Centipede-v5``
* ``ChopperCommand-v5``
* ``Combat-v5``
* ``CrazyClimber-v5``
* ``Crossbow-v5``
* ``Darkchambers-v5``
* ``Defender-v5``
* ``DemonAttack-v5``
* ``DonkeyKong-v5``
* ``DoubleDunk-v5``
* ``Earthworld-v5``
* ``ElevatorAction-v5``
* ``Enduro-v5``
* ``Entombed-v5``
* ``Et-v5``
* ``FishingDerby-v5``
* ``FlagCapture-v5``
* ``Freeway-v5``
* ``Frogger-v5``
* ``Frostbite-v5``
* ``Galaxian-v5``
* ``Gopher-v5``
* ``Gravitar-v5``
* ``Hangman-v5``
* ``HauntedHouse-v5``
* ``Hero-v5``
* ``HumanCannonball-v5``
* ``IceHockey-v5``
* ``Jamesbond-v5``
* ``JourneyEscape-v5``
* ``Joust-v5``
* ``Kaboom-v5``
* ``Kangaroo-v5``
* ``KeystoneKapers-v5``
* ``KingKong-v5``
* ``Klax-v5``
* ``Koolaid-v5``
* ``Krull-v5``
* ``KungFuMaster-v5``
* ``LaserGates-v5``
* ``LostLuggage-v5``
* ``MarioBros-v5``
* ``MazeCraze-v5``
* ``MiniatureGolf-v5``
* ``MontezumaRevenge-v5``
* ``MrDo-v5``
* ``MsPacman-v5``
* ``NameThisGame-v5``
* ``Othello-v5``
* ``Pacman-v5``
* ``Phoenix-v5``
* ``Pitfall-v5``
* ``Pitfall2-v5``
* ``Pong-v5``
* ``Pooyan-v5``
* ``PrivateEye-v5``
* ``Qbert-v5``
* ``Riverraid-v5``
* ``RoadRunner-v5``
* ``Robotank-v5``
* ``Seaquest-v5``
* ``SirLancelot-v5``
* ``Skiing-v5``
* ``Solaris-v5``
* ``SpaceInvaders-v5``
* ``SpaceWar-v5``
* ``StarGunner-v5``
* ``Superman-v5``
* ``Surround-v5``
* ``Tennis-v5``
* ``Tetris-v5``
* ``TicTacToe3d-v5``
* ``TimePilot-v5``
* ``Trondead-v5``
* ``Turmoil-v5``
* ``Tutankham-v5``
* ``UpNDown-v5``
* ``Venture-v5``
* ``VideoCheckers-v5``
* ``VideoChess-v5``
* ``VideoCube-v5``
* ``VideoPinball-v5``
* ``Warlords-v5``
* ``WizardOfWor-v5``
* ``WordZapper-v5``
* ``YarsRevenge-v5``
* ``Zaxxon-v5``
