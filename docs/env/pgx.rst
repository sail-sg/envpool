PGX
===

EnvPool includes native C++ implementations of PGX environments from
`PGX <https://github.com/sotetsuk/pgx>`_ 2.6.0.

Supported Tasks
---------------

EnvPool registers the following PGX tasks:

* ``Go9x9-v1``
* ``Go19x19-v1``
* ``ChineseGo9x9-v1``
* ``ChineseGo19x19-v1``
* ``TicTacToe-v1``
* ``ConnectFour-v1``
* ``Hex-v1``
* ``Othello-v1``
* ``KuhnPoker-v1``
* ``LeducHoldem-v1``
* ``Play2048-v1``
* ``AnimalShogi-v1``
* ``Backgammon-v1``
* ``Chess-v1``
* ``GardnerChess-v1``
* ``Shogi-v1``
* ``SparrowMahjong-v1``

The PGX MinAtar tasks are not registered here because EnvPool already provides
native Atari environments.

Go Rules
--------

``Go9x9-v1`` and ``Go19x19-v1`` follow PGX's Go v1 rules:

* Tromp-Taylor scoring.
* ``N * N + 1`` discrete actions, where the final action is pass.
* Boolean observation shape ``(N, N, 17)`` using AlphaGo Zero history planes.
* SSK legal-action filtering, with positional superko occurrence ending the
  game as a loss for the player who made the repeated position.

EnvPool also provides Chinese-rule variants:

* Chinese area scoring: stones plus empty regions bordered by exactly one
  color, with neutral empty regions counted for neither player.
* Positional superko moves are masked as illegal actions instead of being
  accepted and then turned into a terminal loss.
* The same no-suicide, two-pass terminal, action, and observation API as the
  PGX-compatible Go tasks.

API Notes
---------

PGX turn-based games are exposed through EnvPool's multiplayer API. Each state
contains one observation per player, ``info["current_player"]`` identifies the
player ID whose turn it is, and each environment consumes one action for that
current player.

The task IDs intentionally follow EnvPool style and do not use a ``PGX`` prefix.

Configuration
-------------

Go tasks support ``komi``, ``history_length``, ``max_terminal_steps``, and
``rules``. ``komi`` defaults to ``7.5`` and ``history_length`` defaults to ``8``
to match PGX. ``max_terminal_steps=0`` means ``2 * N * N``. ``rules`` is
``"pgx"`` for the PGX-compatible tasks and ``"chinese"`` for the Chinese-rule
variants.
