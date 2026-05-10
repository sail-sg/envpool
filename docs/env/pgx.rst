PGX
===

EnvPool includes a native C++ implementation of the PGX Go environments from
`PGX <https://github.com/sotetsuk/pgx>`_ 2.6.0.

The default ``go_9x9`` and ``go_19x19`` tasks follow PGX's Go v1 rules:

* Tromp-Taylor scoring.
* Two players.
* ``go_9x9`` and ``go_19x19`` task IDs.
* ``N * N + 1`` discrete actions, where the final action is pass.
* Boolean observation shape ``(N, N, 17)`` using the AlphaGo Zero history
  planes.
* SSK legal-action filtering, with positional superko occurrence ending the
  game as a loss for the player who made the repeated position.

EnvPool also provides Chinese-rule variants:

* Chinese area scoring: stones plus empty regions bordered by exactly one
  color, with neutral empty regions counted for neither player.
* Positional superko moves are masked as illegal actions instead of being
  accepted and then turned into a terminal loss.
* The same no-suicide, two-pass terminal, action, and observation API as the
  PGX-compatible tasks.

Supported Tasks
---------------

* ``go_9x9``; alias: ``PGXGo9x9-v1``
* ``go_19x19``; alias: ``PGXGo19x19-v1``
* ``go_chinese_9x9``; alias: ``ChineseGo9x9-v1``
* ``go_chinese_19x19``; alias: ``ChineseGo19x19-v1``

Observation
-----------

PGX Go is turn-based, but EnvPool exposes it through the existing multiplayer
API. Each state contains two player observations. ``info["current_player"]``
identifies the player ID whose turn it is, and a single action per environment
is interpreted as that player's action.

The Gymnasium observation space is ``MultiBinary((N, N, 17))``. Runtime
observations have leading player dimension ``2`` for each environment and are
returned as boolean arrays.

Info
----

The Gymnasium info dictionary includes:

* ``board``: clipped board values, ``1`` for black, ``-1`` for white, ``0`` for
  empty.
* ``current_player``: player ID to act.
* ``legal_action_mask``: legal actions for the current player.
* ``ko``: SSK ko point, or ``-1``.
* ``is_psk``: whether the latest move produced positional superko.
* ``consecutive_pass_count``.
* ``black_area`` and ``white_area``: area scores before komi under the selected
  rule set.
* ``players.id``.

Configuration
-------------

``komi`` defaults to ``7.5`` and ``history_length`` defaults to ``8`` to match
PGX. ``max_terminal_steps=0`` means ``2 * N * N``, matching PGX's default.
``rules`` is ``"pgx"`` for the PGX-compatible tasks and ``"chinese"`` for the
Chinese-rule variants.
