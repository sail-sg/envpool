# Copyright 2021 Garena Online Private Limited
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
"""VizDoom smoke tests driven by lightweight scripted CV policies."""

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import cast

import cv2
import numpy as np
from absl.testing import absltest

import envpool.vizdoom.registration  # noqa: F401
from envpool.registration import make_gym

_PACKAGE_DIR = os.path.dirname(__file__)
_D1_NUM_ENVS = 4
_D1_MAX_STEPS = 800
_D3_NUM_ENVS = 4
_D3_MAX_STEPS = 600


def _get_map_path(path: str) -> str:
    return os.path.join(_PACKAGE_DIR, "maps", path)


@contextmanager
def _temporary_workdir() -> Iterator[None]:
    prev_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(
        prefix="vizdoom-runtime-", ignore_cleanup_errors=True
    ) as tempdir:
        os.chdir(tempdir)
        try:
            yield
        finally:
            os.chdir(prev_cwd)


def _active_info(info: dict, keep: np.ndarray, done: np.ndarray) -> dict:
    return {
        key: np.asarray(value)[keep]
        for key, value in info.items()
        if np.asarray(value).ndim > 0 and len(np.asarray(value)) == len(done)
    }


# D1 Basic uses EnvPool's combined action space.
_D1_NONE = 0
_D1_TURN_RIGHT = 1
_D1_TURN_LEFT = 2
_D1_FORWARD = 3
_D1_FORWARD_RIGHT = 4
_D1_FORWARD_LEFT = 5
_D1_PICKUP_HEALTH = 68.0
_D1_STAGE_AREA = 180.0


def _d1_medikit(
    frame: np.ndarray,
) -> tuple[float, float, float, float, float] | None:
    gray = frame[..., 0]
    height, width = gray.shape
    mask = (gray > 150).astype(np.uint8) * 255
    mask[: int(height * 0.38), :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((2, 3), np.uint8), iterations=1)
    num, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    best: tuple[float, float, float, float, float, float] | None = None
    for i in range(1, num):
        _, y, w, h, area = stats[i]
        cx, cy = cents[i]
        aspect = w / max(1, h)
        if area < 10 or area > 3500:
            continue
        if y < height * 0.38 or h > height * 0.50 or w > width * 0.75:
            continue
        if (
            aspect < 0.35
            or aspect > 10
            or (w > width * 0.40 and h < height * 0.05)
        ):
            continue
        score = (
            float(area)
            + 0.9 * float(cy)
            + (30.0 if 0.5 < aspect < 6.0 else 0.0)
        )
        candidate = (
            score,
            float(area),
            float(cx),
            float(cy),
            float(w),
            float(h),
        )
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None
    _, area, cx, cy, w, h = best
    return area, cx, cy, w, h


def _d1_action(
    step: int, frame: np.ndarray, health: float, state: dict[str, float]
) -> int:
    height, width = frame.shape[:2]
    medikit = _d1_medikit(frame)
    if medikit is None:
        if step - state.get("last_seen", -999.0) < 40:
            return (
                _D1_TURN_LEFT
                if state.get("lost_spin", -1.0) < 0
                else _D1_TURN_RIGHT
            )
        return (
            _D1_TURN_LEFT
            if (step // 35 + state["id"]) % 2 == 0
            else _D1_TURN_RIGHT
        )

    area, cx, cy, box_w, box_h = medikit
    offset = cx - width / 2
    state["last_seen"] = float(step)
    state["lost_spin"] = -1.0 if offset < 0 else 1.0
    close = (
        area > _D1_STAGE_AREA
        or cy > height * 0.80
        or box_h > height * 0.12
        or box_w > width * 0.20
    )

    if close and health > _D1_PICKUP_HEALTH:
        if offset < -16:
            return _D1_TURN_LEFT
        if offset > 16:
            return _D1_TURN_RIGHT
        return _D1_NONE

    margin = 18 if close else 12
    if offset < -margin:
        return _D1_FORWARD_LEFT
    if offset > margin:
        return _D1_FORWARD_RIGHT
    return _D1_FORWARD


def _eval_d1() -> tuple[np.ndarray, np.ndarray]:
    with _temporary_workdir():
        env = make_gym(
            "D1Basic-v1",
            num_envs=_D1_NUM_ENVS,
            seed=0,
            wad_path=_get_map_path("D1_basic.wad"),
            cfg_path=_get_map_path("D1_basic.cfg"),
            use_combined_action=True,
            stack_num=1,
            frame_skip=1,
            max_episode_steps=_D1_MAX_STEPS,
            render_mode="rgb_array",
            render_width=240,
            render_height=180,
        )
        try:
            rewards = np.zeros(_D1_NUM_ENVS)
            lengths = np.zeros(_D1_NUM_ENVS)
            states = [
                {"id": float(i), "last_seen": -999.0, "lost_spin": -1.0}
                for i in range(_D1_NUM_ENVS)
            ]
            ids = np.arange(_D1_NUM_ENVS)
            _, info = env.reset()
            frames = cast(np.ndarray, env.render(env_ids=ids))
            for step in range(_D1_MAX_STEPS):
                actions = [
                    _d1_action(
                        step,
                        frames[row],
                        float(np.asarray(info["HEALTH"])[row]),
                        states[int(env_id)],
                    )
                    for row, env_id in enumerate(ids)
                ]
                _, rew, terminated, truncated, step_info = env.step(
                    np.asarray(actions, dtype=np.int64), ids
                )
                done = np.logical_or(terminated, truncated)
                cur_ids = np.asarray(step_info["env_id"])
                rewards[cur_ids] += rew
                lengths[cur_ids] += 1
                keep = ~done
                ids = cur_ids[keep]
                info = _active_info(step_info, keep, done)
                if len(ids) == 0:
                    break
                frames = cast(np.ndarray, env.render(env_ids=ids))
            return rewards, lengths
        finally:
            env.close()


def _d3_action(
    attack: int = 0,
    speed: int = 0,
    forward: int = 0,
    back: int = 0,
    right: int = 0,
    left: int = 0,
    turn180: int = 0,
    turn: float = 0.0,
) -> np.ndarray:
    return np.asarray(
        [attack, speed, forward, back, right, left, turn180, turn],
        dtype=np.float64,
    )


@dataclass
class _D3State:
    turn: float = 1.0
    health: float = 100.0
    damage: float = 0.0
    hit: float = 0.0
    dx: float = 0.0
    lock: int = 0
    bad: int = 0
    panic: int = 0
    stuck: int = 0
    last_progress: int = 0
    adapted: bool = False
    bounce_i: int = 0
    prev: np.ndarray | None = None
    recent_views: list[bytes] = field(default_factory=list)
    arc_turn: float | None = None
    bored_after: int | None = None
    bored_turn: int | None = None


def _d3_image(obs: np.ndarray) -> np.ndarray:
    return obs.transpose(1, 2, 0)


def _d3_enemy(frame: np.ndarray) -> tuple[float, float, float, float] | None:
    height, width = frame.shape[:2]
    sx, sy, center = width / 320.0, height / 240.0, width / 2.0
    c0, c1, c2 = [frame[:, :, i].astype(np.int16) for i in range(3)]
    mask = (
        (c0 > 50) & ((c0 - np.maximum(c1, c2)) > 25) & (c1 < 55) & (c2 < 55)
    ).astype(np.uint8) * 255
    mask[: int(40 * sy), :] = 0
    mask[int(205 * sy) :, :] = 0
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((max(2, int(round(4 * sx))),) * 2, np.uint8),
    )
    mask = cv2.dilate(
        mask,
        np.ones((max(2, int(4 * sy)), max(2, int(3 * sx))), np.uint8),
        iterations=1,
    )
    num, _, stats, cents = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    best: tuple[float, float, float, float, float] | None = None
    for i in range(1, num):
        _, _, bw, bh, area = stats[i]
        cx, cy = cents[i]
        if area < 12 * sx * sy or bh < 7 * sy or bw > 95 * sx or bh > 115 * sy:
            continue
        score = area * (bh / (16 * sy)) ** 1.25
        score *= 1 + (cy - 90 * sy) / (180 * sy)
        score /= 1 + abs(cx - center) / (80 * sx)
        if best is None or score > best[0]:
            best = (float(score), float(cx), float(bh), float(area), float(cy))
    if best is None:
        return None
    _, cx, bh, area, cy = best
    return cx, bh, area, cy


def _d3_item(
    frame: np.ndarray, ammo: float = 20.0, hp: float = 100.0
) -> tuple[float, float] | None:
    height, width = frame.shape[:2]
    sx, sy, center = width / 320.0, height / 240.0, width / 2.0
    c0, c1, c2 = [frame[:, :, i].astype(np.int16) for i in range(3)]
    enemy = (c0 > 50) & ((c0 - np.maximum(c1, c2)) > 25) & (c1 < 55) & (c2 < 55)
    item_red = 95 if ammo <= 5 or hp <= -1 else 115
    medikit = (c0 > item_red) & (c1 > 75) & (c2 > 70)
    clip = (c0 > 90) & (c1 > 60) & ((c0 - c2) > 25)
    mask = ((medikit | clip) & (~enemy)).astype(np.uint8) * 255
    mask[: int(50 * sy), :] = 0
    mask[int(218 * sy) :, :] = 0
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((max(2, int(3 * sy)), max(2, int(3 * sx))), np.uint8),
    )
    num, _, stats, cents = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    best: tuple[float, float] | None = None
    for i in range(1, num):
        _, _, bw, bh, area = stats[i]
        cx, cy = cents[i]
        if area < 10 * sx * sy or area > 3000 * sx * sy:
            continue
        if bw > 60 * sx or bh > 52 * sy or bw < 2 * sx or bh < 2 * sy:
            continue
        score = area * (1 + (cy - 85 * sy) / (130 * sy))
        score /= 1 + abs(cx - center) / (85 * sx)
        if best is None or score > best[0]:
            best = (float(score), float(cx))
    if best is not None:
        return best[1], best[0]

    diff = np.maximum.reduce([abs(c0 - c1), abs(c0 - c2), abs(c1 - c2)])
    mask = ((diff > 24) & (~enemy)).astype(np.uint8) * 255
    mask[: int(55 * sy), :] = 0
    mask[int(205 * sy) :, :] = 0
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((max(2, int(3 * sy)), max(2, int(3 * sx))), np.uint8),
    )
    num, _, stats, cents = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    best = None
    for i in range(1, num):
        _, _, bw, bh, area = stats[i]
        cx, cy = cents[i]
        if (
            area < 8 * sx * sy
            or area > 700 * sx * sy
            or bw > 52 * sx
            or bh > 42 * sy
        ):
            continue
        score = area * (1 + (cy - 80 * sy) / (140 * sy))
        score /= 1 + abs(cx - center) / (90 * sx)
        if best is None or score > best[0]:
            best = (float(score), float(cx))
    if best is None:
        return None
    return best[1], best[0]


def _d3_view_hash(frame: np.ndarray) -> bytes:
    gray = frame.astype(np.uint8).mean(axis=2).astype(np.uint8)
    small = cv2.resize(gray, (8, 6), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).astype(np.uint8).tobytes()


def _d3_stuck_turn(state: _D3State) -> float:
    state.bounce_i += 1
    frac = (state.bounce_i * 0.6180339887498949) % 1.0
    state.turn *= -1
    return state.turn * (3.0 + frac * 7.0)


def _d3_choose(
    obs: np.ndarray, info: dict, state: _D3State, step: int
) -> np.ndarray:
    frame = _d3_image(obs)
    height, width = frame.shape[:2]
    sx, sy, center = width / 320.0, height / 240.0, width / 2.0
    ammo = float(info.get("AMMO2", 20.0))
    hp = float(info.get("HEALTH", 100.0))
    damage = float(info.get("DAMAGECOUNT", 0.0))
    hit = float(info.get("HITCOUNT", 0.0))

    if damage > state.damage or hit > state.hit:
        state.lock = 10
        state.bad = 0
        state.last_progress = step
    elif state.lock > 0:
        state.bad += 1
    if hp < state.health - 0.1:
        state.panic = 12
        state.turn *= -1
    state.damage, state.hit, state.health = damage, hit, hp

    enemy = _d3_enemy(frame)
    if enemy is not None and ammo > 0:
        cx, bh, area, _ = enemy
        dx = (cx - center) / sx
        state.dx = dx
        state.lock = max(state.lock, 4)
        turn = float(np.clip(dx * 0.18, -8.0, 8.0))
        if abs(dx) > 4.0:
            danger = area > 220 * sx * sy or bh > 29 * sy
            if danger:
                return _d3_action(
                    speed=1,
                    back=1,
                    right=1 if dx < 0 else 0,
                    left=1 if dx >= 0 else 0,
                    turn=turn,
                )
            return _d3_action(speed=1, forward=1, turn=turn)
        if area > 380 * sx * sy or bh > 42 * sy or state.panic > 0:
            state.panic = max(0, state.panic - 1)
            return _d3_action(
                attack=1,
                speed=1,
                back=1,
                right=1 if (step // 8) % 2 else 0,
                left=0 if (step // 8) % 2 else 1,
                turn=turn,
            )
        return _d3_action(
            attack=1,
            speed=1,
            right=1 if (step // 8) % 2 else 0,
            left=0 if (step // 8) % 2 else 1,
            turn=turn,
        )

    if state.lock > 0 and ammo > 0:
        state.lock -= 1
        turn = float(np.clip(state.dx * 0.18, -8.0, 8.0))
        if abs(state.dx) <= 4.0 and state.bad < 4:
            return _d3_action(attack=1, turn=turn)
        if state.bad >= 4:
            state.turn *= -1
            state.bad = 0
        return _d3_action(
            speed=1, turn=turn if abs(state.dx) > 4.0 else state.turn * 6.0
        )

    if state.panic > 0:
        state.panic -= 1
        return _d3_action(speed=1, back=1, turn=state.turn * 6.0)

    if ammo <= 10:
        item = _d3_item(frame, ammo, hp)
        if item is not None:
            cx, _ = item
            return _d3_action(
                speed=1,
                forward=1,
                turn=float(np.clip(((cx - center) / sx) * 0.18, -8.0, 8.0)),
            )

    view = _d3_view_hash(frame)
    repeated = view in state.recent_views
    state.recent_views.append(view)
    if len(state.recent_views) > 80:
        state.recent_views.pop(0)
    if repeated and step - state.last_progress > 160:
        return _d3_action(speed=1, back=1, turn=_d3_stuck_turn(state))

    if state.prev is not None:
        diff = float(
            np.mean(
                np.abs(frame.astype(np.int16) - state.prev.astype(np.int16))
            )
        )
        state.stuck = state.stuck + 1 if diff < 0.9 else max(0, state.stuck - 1)
    state.prev = frame.copy()
    if state.stuck > 10:
        state.stuck = 0
        state.turn *= -1
        return _d3_action(speed=1, back=1, turn=state.turn * 6.0)

    if not state.adapted and step >= 250:
        low_progress = damage <= 15 and hp >= 90
        danger_progress = hp <= 75 and damage >= 60
        if low_progress or danger_progress:
            state.arc_turn = 1.4
            state.bored_after = 180
            state.bored_turn = 20
        state.adapted = True

    arc_turn = 1.8 if state.arc_turn is None else state.arc_turn
    bored_after = 190 if state.bored_after is None else state.bored_after
    bored_turn = 20 if state.bored_turn is None else state.bored_turn
    bored_arc_turn = arc_turn
    if damage >= 80 and ammo >= 20 and hp >= 85:
        bored_arc_turn = 0.8

    if step - state.last_progress > bored_after:
        phase = (step - state.last_progress - bored_after) % (bored_turn + 90)
        if phase < bored_turn:
            if phase == 0:
                state.turn *= -1
                return _d3_action(speed=1, turn180=1)
            return _d3_action(speed=1, turn=state.turn * 6.0)
        return _d3_action(speed=1, forward=1, turn=state.turn * bored_arc_turn)
    return _d3_action(speed=1, forward=1, turn=state.turn * arc_turn)


def _d3_custom_cfg() -> str:
    with open(_get_map_path("D3_battle.cfg")) as f:
        cfg = f.read()
    cfg = cfg.replace(
        "screen_resolution = RES_160X120", "screen_resolution = RES_320X240"
    )
    cfg = cfg.replace("screen_format = GRAY8", "screen_format = CRCGCB")
    cfg = cfg.replace("render_weapon = true", "render_weapon = false")
    cfg = cfg.replace("render_crosshair = true", "render_crosshair = false")
    buttons = """available_buttons =
    {
        ATTACK
        SPEED
        MOVE_FORWARD
        MOVE_BACKWARD
        MOVE_RIGHT
        MOVE_LEFT
        TURN180
        TURN_LEFT_RIGHT_DELTA
    }

"""
    return (
        cfg[: cfg.index("available_buttons")]
        + buttons
        + cfg[cfg.index("# Game variables") :]
    )


@contextmanager
def _d3_cfg_file(prefix: str = "") -> Iterator[str]:
    fd, path = tempfile.mkstemp(prefix="d3-cv-", suffix=".cfg")
    os.close(fd)
    try:
        with open(path, "w") as f:
            f.write(prefix + _d3_custom_cfg())
        yield path
    finally:
        if os.path.exists(path):
            os.remove(path)


def _eval_d3(
    num_envs: int = _D3_NUM_ENVS,
    cfg_prefix: str = "",
    max_steps: int = _D3_MAX_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    with _temporary_workdir(), _d3_cfg_file(cfg_prefix) as cfg_path:
        env = make_gym(
            "D3Battle-v1",
            num_envs=num_envs,
            seed=0,
            cfg_path=cfg_path,
            wad_path=_get_map_path("D3_battle.wad"),
            use_combined_action=False,
            stack_num=1,
            frame_skip=2,
            max_episode_steps=max_steps,
            img_width=320,
            img_height=240,
            reward_config={"DAMAGECOUNT": [1, 0], "KILLCOUNT": [10, 0]},
            selected_weapon_reward_config={},
        )
        try:
            rewards = np.zeros(num_envs)
            lengths = np.zeros(num_envs)
            states = [_D3State() for _ in range(num_envs)]
            ids = np.arange(num_envs)
            obs, info = env.reset()
            for step in range(max_steps):
                actions = []
                for row, env_id in enumerate(ids):
                    row_info = {
                        key: np.asarray(value)[row]
                        for key, value in info.items()
                        if np.asarray(value).ndim > 0
                        and len(np.asarray(value)) == len(ids)
                    }
                    actions.append(
                        _d3_choose(
                            obs[row], row_info, states[int(env_id)], step
                        )
                    )
                obs2, rew, terminated, truncated, step_info = env.step(
                    np.asarray(actions), ids
                )
                done = np.logical_or(terminated, truncated)
                cur_ids = np.asarray(step_info["env_id"])
                rewards[cur_ids] += rew
                lengths[cur_ids] += 1
                keep = ~done
                ids = cur_ids[keep]
                obs = obs2[keep]
                info = _active_info(step_info, keep, done)
                if len(ids) == 0:
                    break
            return rewards, lengths
        finally:
            env.close()


class _VizdoomPretrainTest(absltest.TestCase):
    def test_0_d3_battle_cv_policy(self) -> None:
        rewards, lengths = _eval_d3()
        self.assertGreaterEqual(rewards.mean(), 150.0)
        self.assertGreaterEqual(rewards.min(), 40.0)
        self.assertGreaterEqual(lengths.mean(), 500)

    def test_1_d1_basic_cv_policy(self) -> None:
        rewards, lengths = _eval_d1()
        self.assertGreaterEqual(rewards.mean(), 0.2)
        self.assertGreaterEqual(rewards.min(), 0.0)
        self.assertGreaterEqual(lengths.mean(), 700)


if __name__ == "__main__":
    absltest.main()
