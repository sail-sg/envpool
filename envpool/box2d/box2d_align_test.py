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
"""Official Gymnasium alignment tests for Box2D environments."""

import importlib.machinery
import importlib.util
import math
import re
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
from absl.testing import absltest

import envpool.box2d.registration as box2d_registration
from envpool.registration import make_gym

assert box2d_registration is not None

_BOX2D_SWIGCONSTANT_RE = re.compile(r"_Box2D\.(\w+_swigconstant)\(")
_SCALE = 30.0


def _patch_box2d_swigconstant_shims(module: Any, pathname: str) -> None:
    wrapper_path = Path(pathname).with_name("Box2D.py")
    try:
        names = set(_BOX2D_SWIGCONSTANT_RE.findall(wrapper_path.read_text()))
    except OSError:
        return
    for attr in names:
        if not hasattr(module, attr):
            setattr(module, attr, lambda _target, _attr=attr: None)


def _install_imp_compat() -> None:
    try:
        import imp  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    compat_imp: Any = types.ModuleType("imp")
    compat_imp.C_EXTENSION = 3

    def find_module(
        name: str, path: Any = None
    ) -> tuple[Any, str, tuple[str, str, int]]:
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.origin is None:
            raise ImportError(name)
        return (
            open(spec.origin, "rb"),
            spec.origin,
            ("", "rb", compat_imp.C_EXTENSION),
        )

    def load_module(
        name: str, file: Any, pathname: str, description: Any
    ) -> Any:
        del file, description
        module = sys.modules.get(name)
        if module is not None:
            return module
        spec = importlib.util.spec_from_file_location(name, pathname)
        if spec is None or spec.loader is None:
            raise ImportError(pathname)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if name == "_Box2D":
            _patch_box2d_swigconstant_shims(module, pathname)
        return module

    compat_imp.find_module = find_module
    compat_imp.load_module = load_module
    sys.modules["imp"] = compat_imp


_install_imp_compat()


def _render_array(env: Any) -> np.ndarray:
    frame = env.render()
    assert frame is not None
    return cast(np.ndarray, frame)


def _unwrap(env: gym.Env[Any, Any]) -> Any:
    return env.unwrapped


def _set_body_state(body: Any, state: np.ndarray) -> None:
    body.position = (float(state[0]), float(state[1]))
    body.angle = float(state[2])
    body.linearVelocity = (float(state[3]), float(state[4]))
    body.angularVelocity = float(state[5])
    body.awake = bool(state[6] > 0.5)


class _UniformReplay:
    def __init__(self, original: Any):
        self._original = original
        self._values: list[float] = []

    def set_values(self, values: np.ndarray) -> None:
        self._values = [float(v) for v in np.ravel(values)]

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if (
            not args
            and not kwargs
            and size is None
            and low == -1.0
            and high == 1.0
            and self._values
        ):
            return self._values.pop(0)
        return self._original.uniform(low, high, size, *args, **kwargs)

    def assert_consumed(self) -> None:
        if self._values:
            raise AssertionError(
                f"unused uniform replay values: {self._values!r}"
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


@contextmanager
def _replay_lunar_uniforms(base: Any, values: np.ndarray) -> Iterator[None]:
    replay = getattr(base, "_envpool_uniform_replay", None)
    if replay is None:
        replay = _UniformReplay(base.np_random)
        base._envpool_uniform_replay = replay
        base._np_random = replay
    replay.set_values(values)
    yield
    replay.assert_consumed()


def _patch_lunar_lander(base: Any, info: dict[str, Any]) -> None:
    import Box2D
    from Box2D.b2 import edgeShape, fixtureDef, polygonShape, revoluteJointDef
    from gymnasium.envs.box2d import lunar_lander

    sky_polys = np.asarray(info["sky_polys"][0], dtype=np.float64)
    base._destroy()
    base.world = Box2D.b2World(gravity=(0, base.gravity))
    base.world.contactListener_keepref = lunar_lander.ContactDetector(base)
    base.world.contactListener = base.world.contactListener_keepref
    base.game_over = False
    base.prev_shaping = None
    base.particles = []

    w = lunar_lander.VIEWPORT_W / lunar_lander.SCALE
    h = lunar_lander.VIEWPORT_H / lunar_lander.SCALE
    chunks = 11
    chunk_x = [w / (chunks - 1) * i for i in range(chunks)]
    base.helipad_x1 = chunk_x[chunks // 2 - 1]
    base.helipad_x2 = chunk_x[chunks // 2 + 1]
    base.helipad_y = h / 4
    base.moon = base.world.CreateStaticBody(
        shapes=edgeShape(vertices=[(0.0, 0.0), (w, 0.0)])
    )
    base.sky_polys = []
    for poly in sky_polys:
        p1 = tuple(float(v) for v in poly[0])
        p2 = tuple(float(v) for v in poly[1])
        base.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
        base.sky_polys.append([
            tuple(float(v) for v in point) for point in poly
        ])
    base.moon.color1 = (0.0, 0.0, 0.0)
    base.moon.color2 = (0.0, 0.0, 0.0)

    base.lander = base.world.CreateDynamicBody(
        position=(w / 2, h),
        angle=0.0,
        fixtures=fixtureDef(
            shape=polygonShape(
                vertices=[
                    (x / lunar_lander.SCALE, y / lunar_lander.SCALE)
                    for x, y in lunar_lander.LANDER_POLY
                ]
            ),
            density=5.0,
            friction=0.1,
            categoryBits=0x0010,
            maskBits=0x001,
            restitution=0.0,
        ),
    )
    base.lander.color1 = (128, 102, 230)
    base.lander.color2 = (77, 77, 128)
    initial_force = np.asarray(info["initial_force"])[0]
    base.lander.ApplyForceToCenter(tuple(float(v) for v in initial_force), True)

    base.legs = []
    for sign in [-1, +1]:
        leg = base.world.CreateDynamicBody(
            position=(
                w / 2 - sign * lunar_lander.LEG_AWAY / lunar_lander.SCALE,
                h,
            ),
            angle=(sign * 0.05),
            fixtures=fixtureDef(
                shape=polygonShape(
                    box=(
                        lunar_lander.LEG_W / lunar_lander.SCALE,
                        lunar_lander.LEG_H / lunar_lander.SCALE,
                    )
                ),
                density=1.0,
                restitution=0.0,
                categoryBits=0x0020,
                maskBits=0x001,
            ),
        )
        leg.ground_contact = False
        leg.color1 = (128, 102, 230)
        leg.color2 = (77, 77, 128)
        rjd = revoluteJointDef(
            bodyA=base.lander,
            bodyB=leg,
            localAnchorA=(0, 0),
            localAnchorB=(
                sign * lunar_lander.LEG_AWAY / lunar_lander.SCALE,
                lunar_lander.LEG_DOWN / lunar_lander.SCALE,
            ),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=lunar_lander.LEG_SPRING_TORQUE,
            motorSpeed=+0.3 * sign,
        )
        if sign == -1:
            rjd.lowerAngle = +0.9 - 0.5
            rjd.upperAngle = +0.9
        else:
            rjd.lowerAngle = -0.9
            rjd.upperAngle = -0.9 + 0.5
        leg.joint = base.world.CreateJoint(rjd)
        base.legs.append(leg)
    base.drawlist = [base.lander] + base.legs
    reset_action = np.array([0, 0]) if base.continuous else 0
    base.step(reset_action)

    _set_body_state(base.lander, np.asarray(info["lander_state"][0]))
    for leg, state in zip(
        base.legs,
        np.asarray(info["leg_states"][0]),
        strict=True,
    ):
        _set_body_state(leg, state)
    for leg, contact in zip(
        base.legs,
        np.asarray(info["ground_contact"][0]),
        strict=True,
    ):
        leg.ground_contact = bool(contact > 0.5)
    base.game_over = bool(np.asarray(info["game_over"])[0] > 0.5)
    base.prev_shaping = float(np.asarray(info["prev_shaping"])[0])
    base.drawlist = [base.lander] + base.legs


def _patch_bipedal_walker(base: Any, info: dict[str, Any]) -> None:
    import Box2D
    from Box2D.b2 import edgeShape, fixtureDef, polygonShape, revoluteJointDef
    from gymnasium.envs.box2d import bipedal_walker

    base._destroy()
    base.world = Box2D.b2World()
    base.world.contactListener_bug_workaround = bipedal_walker.ContactDetector(
        base
    )
    base.world.contactListener = base.world.contactListener_bug_workaround
    base.game_over = False
    base.prev_shaping = None
    base.scroll = 0.0
    base.lidar_render = 0
    base.terrain = []

    path4 = np.asarray(info["path4"][0], dtype=np.float64)
    terrain_polys = path4[:-4] / _SCALE
    for poly in terrain_polys[::-1]:
        body = base.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[tuple(float(v) for v in point) for point in poly]
                ),
                friction=2.5,
            )
        )
        body.color1, body.color2 = (255, 255, 255), (153, 153, 153)
        base.terrain.append(body)

    edge_paths = np.asarray(info["path2"][0], dtype=np.float64) / _SCALE
    base.terrain_poly = []
    for idx, edge in enumerate(edge_paths[::-1]):
        p1 = tuple(float(v) for v in edge[0])
        p2 = tuple(float(v) for v in edge[1])
        body = base.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=edgeShape(vertices=[p1, p2]),
                friction=2.5,
                categoryBits=0x0001,
            )
        )
        color = (76, 255 if idx % 2 == 0 else 204, 76)
        body.color1 = color
        body.color2 = color
        base.terrain.append(body)
        base.terrain_poly.append((
            [p1, p2, (p2[0], 0.0), (p1[0], 0.0)],
            (102, 153, 76),
        ))
    base.terrain.reverse()

    init_x = bipedal_walker.TERRAIN_STEP * bipedal_walker.TERRAIN_STARTPAD / 2
    init_y = bipedal_walker.TERRAIN_HEIGHT + 2 * bipedal_walker.LEG_H
    base.hull = base.world.CreateDynamicBody(
        position=(init_x, init_y), fixtures=bipedal_walker.HULL_FD
    )
    base.hull.color1 = (127, 51, 229)
    base.hull.color2 = (76, 76, 127)
    base.hull.ApplyForceToCenter(
        (float(np.asarray(info["initial_force"])[0]), 0.0), True
    )
    base.legs = []
    base.joints = []
    for sign in [-1, +1]:
        leg = base.world.CreateDynamicBody(
            position=(
                init_x,
                init_y - bipedal_walker.LEG_H / 2 - bipedal_walker.LEG_DOWN,
            ),
            angle=(sign * 0.05),
            fixtures=bipedal_walker.LEG_FD,
        )
        leg.color1 = (153 - sign * 25, 76 - sign * 25, 127 - sign * 25)
        leg.color2 = (102 - sign * 25, 51 - sign * 25, 76 - sign * 25)
        rjd = revoluteJointDef(
            bodyA=base.hull,
            bodyB=leg,
            localAnchorA=(0, bipedal_walker.LEG_DOWN),
            localAnchorB=(0, bipedal_walker.LEG_H / 2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=bipedal_walker.MOTORS_TORQUE,
            motorSpeed=sign,
            lowerAngle=-0.8,
            upperAngle=1.1,
        )
        base.legs.append(leg)
        base.joints.append(base.world.CreateJoint(rjd))

        lower = base.world.CreateDynamicBody(
            position=(
                init_x,
                init_y - bipedal_walker.LEG_H * 3 / 2 - bipedal_walker.LEG_DOWN,
            ),
            angle=(sign * 0.05),
            fixtures=bipedal_walker.LOWER_FD,
        )
        lower.color1 = (153 - sign * 25, 76 - sign * 25, 127 - sign * 25)
        lower.color2 = (102 - sign * 25, 51 - sign * 25, 76 - sign * 25)
        rjd = revoluteJointDef(
            bodyA=leg,
            bodyB=lower,
            localAnchorA=(0, -bipedal_walker.LEG_H / 2),
            localAnchorB=(0, bipedal_walker.LEG_H / 2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=bipedal_walker.MOTORS_TORQUE,
            motorSpeed=1,
            lowerAngle=-1.6,
            upperAngle=-0.1,
        )
        lower.ground_contact = False
        base.legs.append(lower)
        base.joints.append(base.world.CreateJoint(rjd))
    base.drawlist = base.terrain + base.legs + [base.hull]
    base.step(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    clouds = np.asarray(info["cloud_poly"][0], dtype=np.float64)
    base.cloud_poly = [
        (
            [tuple(float(v) for v in point) for point in poly],
            float(np.min(poly[:, 0])),
            float(np.max(poly[:, 0])),
        )
        for poly in clouds
    ]
    base.game_over = bool(np.asarray(info["game_over"])[0] > 0.5)
    base.prev_shaping = float(np.asarray(info["prev_shaping"])[0])
    base.scroll = float(np.asarray(info["scroll"])[0])
    base.drawlist = base.terrain + base.legs + [base.hull]


def _car_road_vertices(
    track: np.ndarray, index: int
) -> list[tuple[float, float]]:
    from gymnasium.envs.box2d import car_racing

    alpha1, beta1, x1, y1 = track[index]
    del alpha1
    alpha2, beta2, x2, y2 = track[index - 1]
    del alpha2
    road1_l = (
        x1 - car_racing.TRACK_WIDTH * math.cos(beta1),
        y1 - car_racing.TRACK_WIDTH * math.sin(beta1),
    )
    road1_r = (
        x1 + car_racing.TRACK_WIDTH * math.cos(beta1),
        y1 + car_racing.TRACK_WIDTH * math.sin(beta1),
    )
    road2_l = (
        x2 - car_racing.TRACK_WIDTH * math.cos(beta2),
        y2 - car_racing.TRACK_WIDTH * math.sin(beta2),
    )
    road2_r = (
        x2 + car_racing.TRACK_WIDTH * math.cos(beta2),
        y2 + car_racing.TRACK_WIDTH * math.sin(beta2),
    )
    return [road1_l, road1_r, road2_r, road2_l]


def _patch_car_racing(base: Any, info: dict[str, Any]) -> None:
    import Box2D
    from Box2D.b2 import fixtureDef, polygonShape
    from gymnasium.envs.box2d import car_racing
    from gymnasium.envs.box2d.car_dynamics import Car

    if base.road:
        for tile in base.road:
            base.world.DestroyBody(tile)
        base.road = []
    if base.car is not None:
        base.car.destroy()

    base.contactListener_keepref = car_racing.FrictionDetector(
        base, base.lap_complete_percent
    )
    base.world = Box2D.b2World(
        (0, 0), contactListener=base.contactListener_keepref
    )
    base.reward = 0.0
    base.prev_reward = 0.0
    base.tile_visited_count = 0
    base.t = 0.0
    base.new_lap = False
    base.track = [
        tuple(float(v) for v in row) for row in np.asarray(info["track"][0])
    ]
    track = np.asarray(base.track, dtype=np.float64)
    base.road = []

    for idx in range(len(track)):
        vertices = _car_road_vertices(track, idx)
        tile = base.world.CreateStaticBody(
            fixtures=fixtureDef(shape=polygonShape(vertices=vertices))
        )
        tile.userData = tile
        color_delta = 0.01 * (idx % 3) * 255
        tile.color = base.road_color + color_delta
        tile.road_visited = False
        tile.road_friction = 1.0
        tile.idx = idx
        tile.fixtures[0].sensor = True
        base.road.append(tile)

    road_poly = np.asarray(info["road_poly"][0], dtype=np.float64)
    road_color = np.asarray(info["road_color"][0], dtype=np.float64)
    base.road_poly = [
        (
            [tuple(float(v) for v in point) for point in poly],
            [int(c) for c in color],
        )
        for poly, color in zip(road_poly, road_color, strict=True)
    ]

    base.car = Car(base.world, *base.track[0][1:4])
    base.car.step(1.0 / car_racing.FPS)
    base.world.Step(1.0 / car_racing.FPS, 6 * 30, 2 * 30)
    base.t += 1.0 / car_racing.FPS

    body_states = np.asarray(info["car_body_states"][0])
    _set_body_state(base.car.hull, body_states[0])
    for wheel, state in zip(base.car.wheels, body_states[1:], strict=True):
        _set_body_state(wheel, state)
    for wheel, state in zip(
        base.car.wheels, np.asarray(info["car_wheel_states"][0]), strict=True
    ):
        wheel.gas = float(state[0])
        wheel.brake = float(state[1])
        wheel.steer = float(state[2])
        wheel.phase = float(state[3])
        wheel.omega = float(state[4])
    base.car.fuel_spent = float(np.asarray(info["car_fuel_spent"])[0])
    base.reward = float(np.asarray(info["car_reward"])[0])
    base.prev_reward = float(np.asarray(info["car_prev_reward"])[0])
    base.t = float(np.asarray(info["car_time"])[0])
    base.new_lap = bool(np.asarray(info["new_lap"])[0] > 0.5)
    base.tile_visited_count = int(np.asarray(info["tile_visited_count"])[0])
    base.state = base._render("state_pixels")


def _official_car_body_states(base: Any) -> np.ndarray:
    bodies = [base.car.hull] + list(base.car.wheels)
    return np.asarray([
        [
            body.position.x,
            body.position.y,
            body.angle,
            body.linearVelocity.x,
            body.linearVelocity.y,
            body.angularVelocity,
            1.0 if body.awake else 0.0,
        ]
        for body in bodies
    ])


def _official_car_wheel_states(base: Any) -> np.ndarray:
    return np.asarray([
        [wheel.gas, wheel.brake, wheel.steer, wheel.phase, wheel.omega]
        for wheel in base.car.wheels
    ])


class Box2DOfficialAlignTest(absltest.TestCase):
    """Check EnvPool Box2D envs against official Gymnasium implementations."""

    def assert_rollout_close(
        self,
        task_id: str,
        env_obs: np.ndarray,
        oracle_obs: np.ndarray,
        env_reward: np.ndarray,
        oracle_reward: float,
        env_term: np.ndarray,
        oracle_term: bool,
        env_trunc: np.ndarray,
        oracle_trunc: bool,
    ) -> None:
        """Assert that one EnvPool step matches one synced official step."""
        obs_atol = 2e-4 if task_id != "CarRacing-v3" else 18.0
        reward_atol = 2e-4
        if task_id == "CarRacing-v3":
            self.assertLess(
                np.abs(
                    env_obs[0].astype(np.int16) - oracle_obs.astype(np.int16)
                ).mean(),
                obs_atol,
            )
        else:
            np.testing.assert_allclose(
                env_obs[0], oracle_obs, rtol=2e-5, atol=obs_atol
            )
        np.testing.assert_allclose(
            env_reward[0], oracle_reward, atol=reward_atol
        )
        self.assertEqual(bool(env_term[0]), oracle_term)
        self.assertEqual(bool(env_trunc[0]), oracle_trunc)

    def test_lunar_lander_rollout_matches_official_after_reset_state_sync(
        self,
    ) -> None:
        """Check LunarLander rollout after reset state synchronization."""
        actions: dict[str, list[Any]] = {
            "LunarLander-v3": [0, 2, 1, 3, 0, 2, 3, 1],
            "LunarLanderContinuous-v3": [
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([0.8, 0.0], dtype=np.float32),
                np.array([-0.2, 0.9], dtype=np.float32),
                np.array([0.5, -0.7], dtype=np.float32),
            ],
        }
        for task_id, action_cycle in actions.items():
            with self.subTest(task_id=task_id):
                env = make_gym(task_id, num_envs=1, seed=3)
                oracle = gym.make(task_id)
                try:
                    env_obs, info = env.reset()
                    oracle.reset(seed=0)
                    base = _unwrap(oracle)
                    _patch_lunar_lander(base, info)
                    for step in range(80):
                        action = action_cycle[step % len(action_cycle)]
                        env_action = np.asarray([action])
                        env_obs, env_rew, env_term, env_trunc, info = env.step(
                            env_action
                        )
                        with _replay_lunar_uniforms(
                            base, np.asarray(info["last_dispersion"][0])
                        ):
                            (
                                oracle_obs,
                                oracle_rew,
                                oracle_term,
                                oracle_trunc,
                                _,
                            ) = oracle.step(action)
                        self.assert_rollout_close(
                            task_id,
                            env_obs,
                            oracle_obs,
                            env_rew,
                            float(oracle_rew),
                            env_term,
                            oracle_term,
                            env_trunc,
                            oracle_trunc,
                        )
                        if bool(env_term[0] or env_trunc[0]):
                            break
                finally:
                    env.close()
                    oracle.close()

    def test_bipedal_walker_rollout_matches_official_after_reset_state_sync(
        self,
    ) -> None:
        """Check BipedalWalker rollout after reset state synchronization."""
        action_cycle = [
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.4, -0.6, -0.2, 0.7], dtype=np.float32),
            np.array([-0.3, 0.5, 0.6, -0.4], dtype=np.float32),
        ]
        for task_id in ("BipedalWalker-v3", "BipedalWalkerHardcore-v3"):
            with self.subTest(task_id=task_id):
                env = make_gym(task_id, num_envs=1, seed=5)
                oracle = gym.make(task_id)
                try:
                    env_obs, info = env.reset()
                    oracle.reset(seed=0)
                    base = _unwrap(oracle)
                    _patch_bipedal_walker(base, info)
                    for step in range(120):
                        action = action_cycle[step % len(action_cycle)]
                        env_obs, env_rew, env_term, env_trunc, info = env.step(
                            np.asarray([action])
                        )
                        (
                            oracle_obs,
                            oracle_rew,
                            oracle_term,
                            oracle_trunc,
                            _,
                        ) = oracle.step(action)
                        self.assert_rollout_close(
                            task_id,
                            env_obs,
                            oracle_obs,
                            env_rew,
                            float(oracle_rew),
                            env_term,
                            oracle_term,
                            env_trunc,
                            oracle_trunc,
                        )
                        if bool(env_term[0] or env_trunc[0]):
                            break
                finally:
                    env.close()
                    oracle.close()

    def test_car_racing_rollout_matches_official_after_reset_state_sync(
        self,
    ) -> None:
        """Check CarRacing rollout after reset state synchronization."""
        action_cycle = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.15, 0.3, 0.0], dtype=np.float32),
            np.array([-0.2, 0.5, 0.0], dtype=np.float32),
            np.array([0.0, 0.2, 0.2], dtype=np.float32),
        ]
        env = make_gym("CarRacing-v3", num_envs=1, seed=7)
        oracle = gym.make("CarRacing-v3", render_mode="state_pixels")
        try:
            env_obs, info = env.reset()
            oracle.reset(seed=0)
            base = _unwrap(oracle)
            _patch_car_racing(base, info)
            for step in range(80):
                action = action_cycle[step % len(action_cycle)]
                env_obs, env_rew, env_term, env_trunc, info = env.step(
                    np.asarray([action])
                )
                (
                    oracle_obs,
                    oracle_rew,
                    oracle_term,
                    oracle_trunc,
                    _,
                ) = oracle.step(action)
                self.assert_rollout_close(
                    "CarRacing-v3",
                    env_obs,
                    oracle_obs,
                    env_rew,
                    float(oracle_rew),
                    env_term,
                    oracle_term,
                    env_trunc,
                    oracle_trunc,
                )
                np.testing.assert_allclose(
                    np.asarray(info["car_body_states"][0]),
                    _official_car_body_states(base),
                    atol=3e-4,
                    rtol=3e-4,
                )
                np.testing.assert_allclose(
                    np.asarray(info["car_wheel_states"][0]),
                    _official_car_wheel_states(base),
                    atol=3e-4,
                    rtol=3e-4,
                )
                if bool(env_term[0] or env_trunc[0]):
                    break
        finally:
            env.close()
            oracle.close()

    def test_reset_render_matches_official_after_state_sync(self) -> None:
        """Check reset render output after official state synchronization."""
        thresholds = {
            "BipedalWalker-v3": 20.0,
            "BipedalWalkerHardcore-v3": 20.0,
            "CarRacing-v3": 18.0,
            "LunarLander-v3": 20.0,
            "LunarLanderContinuous-v3": 20.0,
        }
        for task_id, threshold in thresholds.items():
            with self.subTest(task_id=task_id):
                env = make_gym(
                    task_id,
                    num_envs=1,
                    seed=11,
                    render_mode="rgb_array",
                    render_width=600,
                    render_height=400,
                )
                oracle = gym.make(task_id, render_mode="rgb_array")
                try:
                    _, info = env.reset()
                    oracle.reset(seed=0)
                    base = _unwrap(oracle)
                    if task_id.startswith("LunarLander"):
                        _patch_lunar_lander(base, info)
                    elif task_id.startswith("BipedalWalker"):
                        _patch_bipedal_walker(base, info)
                    else:
                        _patch_car_racing(base, info)
                    frame = _render_array(env)[0].astype(np.int16)
                    expected = np.asarray(oracle.render(), dtype=np.int16)
                    self.assertEqual(frame.shape, expected.shape)
                    self.assertLess(np.abs(frame - expected).mean(), threshold)
                finally:
                    env.close()
                    oracle.close()


if __name__ == "__main__":
    absltest.main()
