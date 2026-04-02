# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared helpers for BabyAI alignment and render tests."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj
from minigrid.envs.babyai.core.verifier import (
    AfterInstr,
    AndInstr,
    BeforeInstr,
    GoToInstr,
    Instr,
    ObjDesc,
    OpenInstr,
    PickupInstr,
    PutNextInstr,
)

import envpool.minigrid.registration  # noqa: F401
from envpool.minigrid import decode_mission
from envpool.registration import list_all_envs

_BABYAI_NUM_TASKS = 96
_OBJ_TYPES = frozenset(("ball", "box", "door", "key"))
_COLOR_NAMES = frozenset(COLOR_NAMES)
_LOC_SUFFIXES = (
    (" in front of you", "front"),
    (" behind you", "behind"),
    (" on your left", "left"),
    (" on your right", "right"),
)
_ALL_STRICT_TASK_IDS = frozenset((
    "BabyAI-OpenDoorDebug-v0",
    "BabyAI-OpenDoorsOrderN2Debug-v0",
    "BabyAI-OpenDoorsOrderN4Debug-v0",
    "BabyAI-PickupDistDebug-v0",
))
_FIRST_OPEN_STRICT_TASK_IDS = frozenset((
    "BabyAI-OpenRedBlueDoorsDebug-v0",
))

_TYPE_NAMES = {
    0: "unseen",
    1: "empty",
    2: "wall",
    3: "floor",
    4: "door",
    5: "key",
    6: "ball",
    7: "box",
    8: "goal",
    9: "lava",
    10: "agent",
}


def babyai_task_ids() -> list[str]:
    task_ids = sorted(
        task_id
        for task_id in list_all_envs()
        if task_id.startswith("BabyAI-")
    )
    if len(task_ids) != _BABYAI_NUM_TASKS:
        raise AssertionError(
            f"expected {_BABYAI_NUM_TASKS} BabyAI tasks, got {len(task_ids)}"
        )
    return task_ids


def mission_from_obs(obs: dict[str, np.ndarray]) -> str:
    mission = obs["mission"]
    arr = np.asarray(mission)
    if arr.ndim == 1:
        return cast(str, decode_mission(arr))
    decoded = cast(np.ndarray, decode_mission(arr))
    return cast(str, decoded[0])


def debug_state(env: Any, env_id: int = 0) -> Any:
    debug_states = cast(Any, env)._debug_states(
        np.asarray([env_id], dtype=np.int32)
    )
    return debug_states[0]


def patch_render_state(env: Any, debug_state: Any, elapsed_step: int) -> None:
    env.grid = _rebuild_grid(debug_state)
    env.agent_pos = tuple(debug_state.agent_pos)
    env.agent_dir = int(debug_state.agent_dir)
    env.mission = debug_state.mission
    env.carrying = _rebuild_carrying(debug_state)
    env.max_steps = int(debug_state.max_steps)
    env.step_count = int(elapsed_step)


def patch_verifier_state(
    env: Any,
    task_id: str,
    debug_state: Any,
    elapsed_step: int,
) -> None:
    patch_render_state(env, debug_state, elapsed_step)
    env.instrs = _parse_instr(task_id, env.mission)
    env.instrs.reset_verifier(env)
    _attach_carrying_to_put_next(env.instrs, env.carrying)


def _decode_obj(type_idx: int, color_idx: int, state: int) -> WorldObj | None:
    return WorldObj.decode(int(type_idx), int(color_idx), int(state))


def _rebuild_grid(debug_state: Any) -> Grid:
    width = int(debug_state.width)
    height = int(debug_state.height)
    grid = Grid(width, height)
    grid_data = np.asarray(debug_state.grid, dtype=np.uint8).reshape((
        width,
        height,
        3,
    ))
    contains_data = np.asarray(
        debug_state.grid_contains,
        dtype=np.uint8,
    ).reshape((width, height, 3))
    for x in range(width):
        for y in range(height):
            obj = _decode_obj(*grid_data[x, y])
            if obj is not None:
                contains = _decode_obj(*contains_data[x, y])
                if contains is not None:
                    obj.contains = contains
                obj.init_pos = (x, y)
                obj.cur_pos = (x, y)
            grid.set(x, y, obj)
    return grid


def _rebuild_carrying(debug_state: Any) -> WorldObj | None:
    if not debug_state.has_carrying:
        return None
    carrying = _decode_obj(
        int(debug_state.carrying_type),
        int(debug_state.carrying_color),
        int(debug_state.carrying_state),
    )
    assert carrying is not None
    if debug_state.carrying_has_contains:
        contains = _decode_obj(
            int(debug_state.carrying_contains_type),
            int(debug_state.carrying_contains_color),
            int(debug_state.carrying_contains_state),
        )
        if contains is not None:
            carrying.contains = contains
    carrying.init_pos = (-1, -1)
    carrying.cur_pos = (-1, -1)
    return carrying


def _parse_instr(task_id: str, mission: str) -> Instr:
    instr = _parse_instr_text(mission.strip())
    _apply_task_strictness(task_id, instr)
    return instr


def _parse_instr_text(text: str) -> Instr:
    if ", then " in text:
        instr_a, instr_b = text.split(", then ", 1)
        return BeforeInstr(
            _parse_instr_text(instr_a),
            _parse_instr_text(instr_b),
        )
    if " after you " in text:
        instr_a, instr_b = text.split(" after you ", 1)
        return AfterInstr(
            _parse_instr_text(instr_a),
            _parse_instr_text(instr_b),
        )
    if " and " in text:
        instr_a, instr_b = text.split(" and ", 1)
        return AndInstr(
            _parse_instr_text(instr_a),
            _parse_instr_text(instr_b),
        )
    if text.startswith("go to "):
        return GoToInstr(_parse_obj_desc(text[len("go to ") :]))
    if text.startswith("open "):
        return OpenInstr(_parse_obj_desc(text[len("open ") :]))
    if text.startswith("pick up "):
        return PickupInstr(_parse_obj_desc(text[len("pick up ") :]))
    if text.startswith("put "):
        desc_move, desc_fixed = text[len("put ") :].split(" next to ", 1)
        return PutNextInstr(
            _parse_obj_desc(desc_move),
            _parse_obj_desc(desc_fixed),
        )
    raise AssertionError(f"unsupported BabyAI mission: {text}")


def _parse_obj_desc(text: str) -> ObjDesc:
    loc = None
    text = text.strip()
    for suffix, suffix_loc in _LOC_SUFFIXES:
        if text.endswith(suffix):
            loc = suffix_loc
            text = text[: -len(suffix)]
            break
    for article in ("a ", "the "):
        if text.startswith(article):
            text = text[len(article) :]
            break

    tokens = text.split()
    if len(tokens) == 1:
        color = None
        obj_type = None if tokens[0] == "object" else tokens[0]
    elif len(tokens) == 2:
        color = tokens[0]
        obj_type = None if tokens[1] == "object" else tokens[1]
    else:
        raise AssertionError(f"unsupported BabyAI object description: {text}")

    if color is not None and color not in _COLOR_NAMES:
        raise AssertionError(f"unsupported BabyAI object color: {text}")
    if obj_type is not None and obj_type not in _OBJ_TYPES:
        raise AssertionError(f"unsupported BabyAI object type: {text}")
    return ObjDesc(obj_type, color, loc)


def _apply_task_strictness(task_id: str, instr: Instr) -> None:
    if task_id in _ALL_STRICT_TASK_IDS:
        _set_instr_strict(instr)
    elif task_id in _FIRST_OPEN_STRICT_TASK_IDS and isinstance(
        instr, BeforeInstr
    ):
        _set_instr_strict(instr.instr_a)


def _set_instr_strict(instr: Instr) -> None:
    if isinstance(instr, (OpenInstr, PickupInstr, PutNextInstr)):
        instr.strict = True
    if isinstance(instr, (AfterInstr, AndInstr, BeforeInstr)):
        _set_instr_strict(instr.instr_a)
        _set_instr_strict(instr.instr_b)


def _attach_carrying_to_put_next(
    instr: Instr,
    carrying: WorldObj | None,
) -> None:
    if carrying is None:
        return
    if isinstance(instr, PutNextInstr):
        if _matches_obj_desc(carrying, instr.desc_move):
            if carrying not in instr.desc_move.obj_set:
                instr.desc_move.obj_set.append(carrying)
    elif isinstance(instr, (AfterInstr, AndInstr, BeforeInstr)):
        _attach_carrying_to_put_next(instr.instr_a, carrying)
        _attach_carrying_to_put_next(instr.instr_b, carrying)


def _matches_obj_desc(obj: WorldObj, desc: ObjDesc) -> bool:
    if desc.type is not None and obj.type != desc.type:
        return False
    if desc.color is not None and obj.color != desc.color:
        return False
    return True
