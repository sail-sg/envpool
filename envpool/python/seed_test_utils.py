# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Behavior checks shared by environment seeding tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

import numpy as np

from envpool.registration import make_gymnasium, registry

_NUM_ENVS = 2
_SEEDS = (11, 11, 43)


def registered_task_ids(module: str) -> list[str]:
    """Return a family's registered tasks, deduplicating identical aliases."""
    tasks = []
    seen = set()
    for task_id, (owner, spec, config) in sorted(registry.specs.items()):
        if owner != module:
            continue
        key = (spec, json.dumps(config, sort_keys=True, default=str))
        if key not in seen:
            seen.add(key)
            tasks.append(task_id)
    return tasks


def _fingerprint(value: Any, slot: int | None = None) -> bytes:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            array = np.asarray(item)
            if slot is not None and array.ndim:
                # EnvPool groups player observations by environment. Preserve
                # all players in each group, rather than comparing players.
                count = len(array) // _NUM_ENVS
                array = array[slot * count : (slot + 1) * count]
            digest.update(str((array.dtype.str, array.shape)).encode())
            if array.dtype.hasobject:
                digest.update(repr(array.tolist()).encode())
            else:
                digest.update(np.ascontiguousarray(array).tobytes())

    visit(value)
    return digest.digest()


def _state(obs: Any, info: dict[str, Any], info_keys: tuple[str, ...]) -> Any:
    # Do not include env_id, episode counters, engine_seed, or RNG state:
    # changing these does not establish that the simulated state changes.
    return {"obs": obs, **{key: info[key] for key in info_keys}}


def _make_pools(
    stack: ExitStack, task_id: str, kwargs: dict[str, Any]
) -> list[Any]:
    pools = []
    for seed in _SEEDS:
        pool = make_gymnasium(
            task_id, num_envs=_NUM_ENVS, num_threads=1, seed=seed, **kwargs
        )
        stack.callback(pool.close)
        pools.append(pool)
    return pools


def check_seeded_resets(
    test: Any,
    task_id: str,
    *,
    info_keys: tuple[str, ...] = (),
    expected: tuple[bool | None, bool | None, bool | None] = (True, True, True),
    field_expectations: dict[str, tuple[bool | None, bool | None, bool | None]]
    | None = None,
    extra_state: Callable[[Any], Any] | None = None,
    **kwargs: Any,
) -> None:
    """Check reproducibility and seed/reset/parallel state variation."""
    sequences: list[list[bytes]] = [[], [], []]
    parallel_differs = False
    field_sequences: dict[str, list[list[bytes]]] = {
        key: [[], [], []] for key in field_expectations or {}
    }
    field_parallel = dict.fromkeys(field_sequences, False)
    with ExitStack() as stack:
        pools = _make_pools(stack, task_id, kwargs)
        for reset_index in range(8):
            states = []
            for pool_index, (pool, sequence) in enumerate(
                zip(pools, sequences, strict=True)
            ):
                obs, info = pool.reset()
                state = _state(obs, info, info_keys)
                if extra_state is not None:
                    state["hidden_state"] = extra_state(pool)
                states.append(_fingerprint(state))
                sequence.append(states[-1])
                parallel_differs |= _fingerprint(state, 0) != _fingerprint(
                    state, 1
                )
                for key, traces in field_sequences.items():
                    traces[pool_index].append(_fingerprint(state[key]))
                    field_parallel[key] |= _fingerprint(
                        state[key], 0
                    ) != _fingerprint(state[key], 1)
            test.assertEqual(
                states[0],
                states[1],
                f"{task_id}: same seed differs at reset {reset_index}",
            )
    actual = (
        sequences[0] != sequences[2],
        any(len(set(sequence)) > 1 for sequence in sequences),
        parallel_differs,
    )
    for name, value, wanted in zip(
        (
            "different seeds change state",
            "successive resets change state",
            "parallel environments differ",
        ),
        actual,
        expected,
        strict=True,
    ):
        if wanted is not None:
            with test.subTest(property=name):
                test.assertEqual(value, wanted, f"{task_id}: {name}")
    for key, wanted in (field_expectations or {}).items():
        traces = field_sequences[key]
        actual_field = (
            traces[0] != traces[2],
            any(len(set(trace)) > 1 for trace in traces),
            field_parallel[key],
        )
        for index, requirement in enumerate(wanted):
            if requirement is not None:
                with test.subTest(field=key, property=index):
                    test.assertEqual(
                        actual_field[index],
                        requirement,
                        f"{task_id}: {key} seed/reset/parallel variation {index}",
                    )


def _sample_action(pool: Any, obs: Any, info: dict[str, Any], rng: Any) -> Any:
    space = pool.action_space
    mask = info.get("legal_action_mask")
    if mask is None and isinstance(obs, dict):
        mask = obs.get("action_mask")
    if mask is not None:
        mask = np.asarray(mask)[0]
    if (
        hasattr(space, "n")
        and mask is not None
        and mask.ndim == 1
        and mask.any()
    ):
        action = rng.choice(np.flatnonzero(mask))
    elif hasattr(space, "nvec") and mask is not None and mask.any():
        if tuple(space.nvec) == mask.shape:
            action = np.asarray(
                np.unravel_index(rng.choice(np.flatnonzero(mask)), mask.shape)
            )
        elif mask.ndim == 2 and mask.shape[0] == len(space.nvec):
            action = np.asarray([
                rng.choice(np.flatnonzero(row)) if row.any() else 0
                for row in mask
            ])
        else:
            action = space.sample()
    else:
        action = space.sample()
    count = len(info.get("players", {}).get("env_id", info["env_id"]))
    dtype = pool.spec.action_array_spec["action"].dtype
    # Identical actions across seeds AND vector slots ensure any divergence
    # comes from environment randomness, not independently sampled actions.
    return np.repeat(np.asarray(action, dtype=dtype)[None], count, axis=0)


def check_seeded_rollouts(
    test: Any,
    task_id: str,
    *,
    info_keys: tuple[str, ...] = (),
    expect_different: bool = True,
    **kwargs: Any,
) -> None:
    """Exercise randomness that is hidden until actions are taken."""
    differs = False
    with ExitStack() as stack:
        pools = _make_pools(stack, task_id, kwargs)
        pools[0].action_space.seed(123)
        rng = np.random.default_rng(123)
        for episode in range(3):
            resets = [pool.reset() for pool in pools]
            obs, info = resets[0]
            for step_index in range(64):
                action = _sample_action(pools[0], obs, info, rng)
                steps = [pool.step(action) for pool in pools]
                fingerprints = [
                    _fingerprint((
                        _state(step[0], step[4], info_keys),
                        step[1:4],
                    ))
                    for step in steps
                ]
                test.assertEqual(
                    fingerprints[0],
                    fingerprints[1],
                    f"{task_id}: same seed differs in episode {episode}, step {step_index}",
                )
                differs |= fingerprints[0] != fingerprints[2]
                obs, info = steps[0][0], steps[0][4]
    test.assertEqual(
        differs, expect_different, f"{task_id}: seed sensitivity during rollout"
    )
