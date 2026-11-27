# Copyright 2026 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Behavioral alignment with Gymnax 1.1.0, including raw episode endings."""

from collections import deque
from contextlib import closing
from typing import Any

import gymnax
import jax
import numpy as np
from absl.testing import absltest, parameterized

import envpool.classic_control.registration  # noqa: F401
import envpool.gymnax.registration  # noqa: F401
from envpool.gymnax.oracle import (
    first_slot,
    frame,
    joint_state,
    make_oracle,
    renderable,
    stepper,
    synchronize,
)
from envpool.registration import make_dm, make_gymnasium, registry

_IDS = gymnax.registered_envs
_CASES = [(name.replace("-", "_"), name) for name in _IDS]


def _toward_goal(walkable: np.ndarray, start: int, goal: int) -> int:
    cols = walkable.shape[1]
    frontier = deque([(start, -1)])
    seen = {start}
    while frontier:
        pos, first = frontier.popleft()
        for a, offset in enumerate((-cols, 1, cols, -1)):
            neighbor = pos + offset
            if not walkable.flat[neighbor]:
                continue
            move = a if first < 0 else first
            if neighbor == goal:
                return move
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append((neighbor, move))
    raise AssertionError("No path to goal")


def _asterix(obs: np.ndarray) -> int:
    y, x = np.argwhere(obs[:, :, 0])[0]
    gold = np.argwhere(obs[:, :, 3])
    candidates = []
    for a, (r, c) in enumerate((
        (y, x),
        (y, x - 1),
        (y - 1, x),
        (y, x + 1),
        (y + 1, x),
    )):
        if 1 <= r <= 8 and 2 <= c <= 7:
            danger = obs[r, c - 1 : c + 2, 1].any()
            distance = (
                np.abs(gold - [r, c]).sum(axis=1).min() if len(gold) else 0
            )
            candidates.append((danger, distance, abs(c - 5), a))
    return min(candidates)[-1]


def _seaquest(obs: np.ndarray, t: int) -> int:
    # Engage the first enemy from its lane, then recover a diver and surface.
    y, x = np.argwhere(obs[:, :, 0])[0]
    enemies = np.argwhere(obs[:, :, 5:7].any(axis=-1))
    if t < 21:
        return 0
    if t < 36 and len(enemies):
        r, c = enemies[0]
        if t == 21:
            return 1 if c < x else 3
        if r != y:
            return 4
        return 5
    if obs[:, :, 8].any():
        return 2
    divers = np.argwhere(obs[:, :, 9])
    if len(divers):
        r, c = divers[0]
        if c != x:
            return 1 if c < x else 3
        return 2 if r < y else 4
    return 5


def _action(
    env: Any,
    start: np.ndarray,
    obs: np.ndarray,
    info: dict,
    t: int,
    swing: bool = False,
) -> Any:
    name = env.name.split("-")[0]
    h: Any = info.get("hidden")
    if name == "CartPole":
        return int(t % 3 == 1)
    if name == "Pendulum":
        return [np.copysign(2, obs[2]) if swing else np.clip(-obs[2], -2, 2)]
    if name == "Acrobot":
        return (2 if obs[5] >= 0 else 0) if swing else t % 3
    if name == "MountainCar":
        return 2 if obs[1] >= 0 else 0
    if name == "MountainCarContinuous":
        return [1 if obs[1] >= 0 else -1]
    if name == "PointRobot":
        return np.clip(h[:2] - obs[:2], -0.15, 0.15)
    if name == "Reacher":
        return np.sin([t / 5, t / 9])
    if name == "Swimmer":
        target = obs[14:] if t % 100 < 50 else obs[:2]
        return np.clip(target - obs[10:12] - obs[12:14], -1, 1)
    if name == "Catch":
        return 1 + np.sign(start[0].argmax() - obs[-1].argmax())
    if name == "DeepSea":
        return int(env.action_mapping[t, t])
    if name == "DiscountingChain":
        return 4 if t == 0 else t % 5
    if name == "MemoryChain":
        return int(start[2] > 0) if t == 5 else t % 2
    if name == "UmbrellaChain":
        return int(start[0]) if t == 0 else t % 2
    if name == "MNISTBandit":
        return int(env.labels[int(h[0])])
    if name == "SimpleBandit":
        return h.argmax()
    if name == "MetaMaze":
        return _toward_goal(np.asarray(env.env_map), int(h[1]), int(h[0]))
    if name == "FourRooms":
        return _toward_goal(
            np.asarray(env.env_map), 13 * obs[0] + obs[1], 13 * obs[2] + obs[3]
        )
    if name == "Snake":
        y, x = np.argwhere(obs[:, :, 1])[0]
        if env.height == 2:
            return ((1, 2), (0, 3))[y][x]
        r, c = np.argwhere(obs[:, :, 2])[0]
        return (0 if r < y else 2) if r != y else (1 if c > x else 3)
    if name == "Freeway":
        return 2 if t % 43 == 0 else 1
    if name == "Breakout":
        delta = np.argwhere(obs[:, :, 1])[0, 1] - obs[9, :, 0].argmax()
        return 1 if delta < 0 else int(delta > 0) * 2
    if name == "SpaceInvaders":
        return 3 if t % 5 == 0 else (1 if t % 20 < 10 else 2)
    if name == "Pong":
        return 1 if t in (40, 41) else int(t >= 75) * 2
    if name == "Asterix":
        return _asterix(obs)
    if name == "Seaquest":
        return _seaquest(obs, t)
    return t % env.num_actions


def _check_frame(pool: Any, reference: Any, params: Any, state: Any) -> None:
    actual = pool.render()[0]
    expected = frame(reference, params, state, *actual.shape[:2])
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.uint8
    error = np.abs(actual.astype(np.int16) - expected)
    name = reference.name
    if name.endswith("-MinAtar") or name == "Pong-misc":
        # Discrete cells and channel priority must match across the whole board.
        assert error.max() <= 1
    elif name in {"Catch-bsuite", "FourRooms-misc", "MetaMaze-misc"}:
        # Different fonts, but identical wall, open and marked cells.
        def tiles(img: np.ndarray) -> np.ndarray:
            h, w = img.shape[:2]
            return img.reshape(h // 40, 40, w // 40, 40, 3).mean((1, 3, 4))

        for shade in (1, 254):
            np.testing.assert_array_equal(
                tiles(actual) < shade, tiles(expected) < shade
            )
        assert tiles(error).max() < 40
    else:
        # The existing classic-control budget includes edge antialiasing.
        assert error.mean() < 4
        if name == "PointRobot-misc":
            # Both small colored disks must coincide even on a mostly white canvas.
            for high, low in ((0, 2), (2, 0)):
                a, b = [
                    img[:, :, high].astype(int) - img[:, :, low] > 20
                    for img in (actual, expected)
                ]
                assert (a & b).sum() >= 0.7 * (a | b).sum()


class _GymnaxTest(parameterized.TestCase):
    def test_registry(self) -> None:
        self.assertSetEqual(
            {"Gymnax/" + name for name in _IDS},
            {s for s in registry.specs if s.startswith("Gymnax/")},
        )
        for name in _IDS:
            if "-v" in name:
                self.assertEqual(
                    registry.specs[name][0], "envpool.classic_control"
                )
            else:
                self.assertEqual(
                    registry.specs[name], registry.specs["Gymnax/" + name]
                )

    def _rollout(
        self,
        name: str,
        *,
        seed: int = 7,
        actions: list[int] | None = None,
        swing: bool = False,
        **options: Any,
    ) -> tuple[float, bool, bool]:
        ref, p = make_oracle(name, **options)
        with closing(
            make_gymnasium(
                "Gymnax/" + name,
                seed=seed,
                render_mode="rgb_array" if renderable(name) else None,
                **options,
            )
        ) as pool:
            self.assertEqual(
                pool.config["max_episode_steps"], p.max_steps_in_episode
            )
            act = ref.action_space(p)
            space = pool.action_space
            if hasattr(act, "n"):
                self.assertEqual(space.n, act.n)
            else:
                self.assertEqual(space.shape, act.shape)
                np.testing.assert_allclose(space.low, act.low)
                np.testing.assert_allclose(space.high, act.high)
            ob, data = pool.reset()
            start = obs = ob[0]
            info = first_slot(data)
            self.assertTrue(pool.observation_space.contains(obs))
            state = synchronize(ref, p, obs, info)
            advance = stepper(ref, p)
            if renderable(name):
                _check_frame(pool, ref, p, state)
            peak = -np.inf
            # Full low-energy and five-step swing trajectories stayed within
            # 1.4e-5 for obs/reward across 25 resets (JAX f32, native f64).
            physics = "-v" in name or not hasattr(act, "n")
            atol, rtol = (2e-5, 1e-5) if physics else (2e-7, 1e-6)
            for t in range(int(p.max_steps_in_episode)):
                choice = (
                    actions[t % len(actions)]
                    if actions
                    else _action(ref, start, obs, info, t, swing)
                )
                a = np.asarray(
                    choice, dtype=pool.spec.action_array_spec["action"].dtype
                )
                ob, rew, term, trunc, data = pool.step(a[None])
                obs, info = ob[0], first_slot(data)
                o, state, r, end, limit, extra = advance(state, a, obs, info)
                o, r, end, limit, extra = jax.device_get((
                    o,
                    r,
                    end,
                    limit,
                    extra,
                ))
                where = f"{name} step {t + 1}"
                self.assertEqual(obs.shape, o.shape, where)
                self.assertEqual(obs.dtype, o.dtype, where)
                self.assertTrue(pool.observation_space.contains(obs), where)
                self.assertEqual(rew.dtype, np.float32)
                self.assertEqual(term.dtype, np.bool_)
                self.assertEqual(trunc.dtype, np.bool_)
                self.assertEqual((term[0], trunc[0]), (end, limit), where)
                if obs.ndim >= 2 or obs.dtype.kind == "i":
                    np.testing.assert_array_equal(obs, o, err_msg=where)
                else:
                    np.testing.assert_allclose(
                        obs, o, atol=atol, rtol=rtol, err_msg=where
                    )
                np.testing.assert_allclose(
                    rew[0], r, atol=atol, rtol=rtol, err_msg=where
                )
                for key in ("p", "length"):
                    if key in extra:
                        self.assertEqual(info[key], extra[key], where)
                if name == "MetaMaze-misc":
                    goal, pos = info["hidden"][:2].astype(int)
                    self.assertTrue(np.asarray(ref.env_map).flat[pos])
                    np.testing.assert_array_equal(divmod(pos, 11), state.pos)
                    np.testing.assert_array_equal(divmod(goal, 11), state.goal)
                elif name == "PointRobot-misc" and r > 0:
                    self.assertGreaterEqual(obs[1], 0)
                    self.assertLessEqual(np.linalg.norm(obs[:2]), 1)
                    np.testing.assert_array_equal(
                        info["hidden"][:2], state.goal
                    )
                elif name == "Snake-misc" and r > 0:
                    if end:
                        self.assertEqual(info["length"], ref.width * ref.height)
                    else:
                        self.assertFalse(obs[obs[:, :, 2] == 1, :2].any())
                if name == "Acrobot-v1":
                    np.testing.assert_allclose(
                        info["state"],
                        [state.joint_angle1, state.joint_angle2],
                        atol=atol,
                        rtol=rtol,
                        err_msg=where,
                    )
                peak = max(peak, float(rew[0]))
                if renderable(name) and (t in (3, 30) or end or limit):
                    _check_frame(pool, ref, p, state)
                if end or limit:
                    # EnvPool returns the final observation before the next-step reset.
                    _, rew, term, trunc, data = pool.step(a[None])
                    self.assertEqual(data["elapsed_step"][0], 0)
                    self.assertEqual(
                        (rew[0], term[0], trunc[0]), (0, False, False)
                    )
                    return peak, bool(end), bool(limit)
                if swing and (t + 1) % 5 == 0:
                    state = joint_state(name, state, obs, info)
        raise AssertionError(f"{name} exceeded the episode limit")

    @parameterized.named_parameters(_CASES)
    def test_oracle(self, name: str) -> None:
        peak, ended, _ = self._rollout(name)
        if name.endswith(("-bsuite", "-MinAtar")) or name in {
            "Snake-misc",
            "PointRobot-misc",
            "MetaMaze-misc",
            "FourRooms-misc",
            "MountainCarContinuous-v0",
        }:
            self.assertGreater(peak, 0)
        if name == "MountainCar-v0":
            self.assertTrue(ended)

    def test_swing_up(self) -> None:
        # These fast rotations amplify rounding. Reanchor only after comparing
        # the entire segment. test_oracle also checks uninterrupted episodes.
        self.assertEqual(
            self._rollout("Acrobot-v1", swing=True), (0, True, False)
        )
        self._rollout("Pendulum-v1", swing=True)

    def test_frozen_lake_goal_and_hole(self) -> None:
        self.assertEqual(
            self._rollout(
                "FrozenLake-misc", is_slippery=False, actions=[1, 1, 2, 1, 2, 2]
            ),
            (1, True, False),
        )
        self.assertEqual(
            self._rollout("FrozenLake-misc", is_slippery=False, actions=[1, 2]),
            (0, True, False),
        )

    def test_snake_full_board(self) -> None:
        self.assertEqual(
            self._rollout("Snake-misc", width=2, height=2), (1, True, False)
        )

    def test_chain_choices(self) -> None:
        ref, _ = make_oracle("DeepSea-bsuite")
        left = 1 - int(ref.action_mapping[0, 0])
        self.assertEqual(
            self._rollout(ref.name, actions=[left, 1 - left]), (0, True, True)
        )
        # Later choices cannot change which step pays out.
        with closing(
            make_gymnasium("DiscountingChain-bsuite", num_envs=5)
        ) as pool:
            pool.reset()
            actions = np.arange(5, dtype=np.int32)
            rewards = np.stack([
                pool.step((actions + t) % 5)[1] for t in range(100)
            ])
            np.testing.assert_array_equal(
                rewards.argmax(axis=0) + 1, [1, 3, 10, 30, 100]
            )
            np.testing.assert_allclose(rewards.sum(axis=0), [1.1, 1, 1, 1, 1])

    def test_time_limit(self) -> None:
        for name, action, natural in (
            ("CartPole-v1", 0, False),
            ("Catch-bsuite", 1, False),
            ("SimpleBandit-bsuite", 1, True),
            ("MNISTBandit-bsuite", 2, True),
        ):
            with self.subTest(name=name):
                with closing(
                    make_gymnasium("Gymnax/" + name, max_episode_steps=1)
                ) as pool:
                    pool.reset()
                    _, _, term, trunc, _ = pool.step(np.array([action]))
                    self.assertEqual((term[0], trunc[0]), (natural, True))
                with closing(
                    make_dm("Gymnax/" + name, max_episode_steps=1)
                ) as dm:
                    dm.reset()
                    last = dm.step(np.array([action]))
                    self.assertTrue(last.last())
                    self.assertEqual(last.discount[0], 0 if natural else 1)

    @parameterized.named_parameters([
        case for case in _CASES if renderable(case[1])
    ])
    def test_render_selection(self, name: str) -> None:
        def create() -> Any:
            return make_gymnasium(
                "Gymnax/" + name,
                num_envs=3,
                seed=11,
                render_mode="rgb_array",
                render_width=96,
                render_height=64,
            )

        with closing(create()) as pool, closing(create()) as other:
            if "-v" not in name:
                with self.assertRaisesRegex(
                    RuntimeError, "reset before rendering"
                ):
                    pool.render(env_ids=[2])
            pool.reset()
            other.reset()
            pool.action_space.seed(23)
            for t in range(33):
                if t in (0, 4, 32):
                    frames = pool.render(env_ids=[0, 1, 2])
                    self.assertEqual(frames.shape, (3, 64, 96, 3))
                    self.assertEqual(frames.dtype, np.uint8)
                    np.testing.assert_array_equal(
                        frames[[2, 0]], other.render(env_ids=[2, 0])
                    )
                a = np.array([pool.action_space.sample() for _ in range(3)])
                pool.step(a)
                other.step(a)

    @parameterized.named_parameters([
        case for case in _CASES if case[1].endswith("-MinAtar")
    ])
    def test_full_action_set(self, name: str) -> None:
        ref, p = make_oracle(name, use_minimal_action_set=False)
        minimal = list(np.asarray(ref.minimal_action_set))
        with (
            closing(make_gymnasium(name)) as compact,
            closing(make_gymnasium(name, use_minimal_action_set=False)) as full,
        ):
            self.assertEqual(full.action_space.n, ref.action_space(p).n)
            compact.reset()
            full.reset()
            for t in range(64):
                a = t % full.action_space.n
                b = minimal.index(a) if a in minimal else 0
                np.testing.assert_equal(
                    full.step(np.array([a])), compact.step(np.array([b]))
                )

    def test_continuous_mountain_car_wall(self) -> None:
        name = "MountainCarContinuous-v0"
        with (
            closing(make_gymnasium(name)) as classic,
            closing(make_gymnasium("Gymnax/" + name)) as gymnax_car,
        ):
            obs, _ = classic.reset()
            gymnax_car.reset()
            for _ in range(200):
                action = np.copysign(1, obs[:, 1:])
                obs = classic.step(action)[0]
                other = gymnax_car.step(action)[0]
                if other[0, 0] == np.float32(-1.2):
                    self.assertLess(other[0, 1], 0)
                    self.assertEqual(obs[0, 1], 0)
                    break
                np.testing.assert_array_equal(obs, other)
            else:
                self.fail("Trajectory missed the left wall")


if __name__ == "__main__":
    absltest.main()
