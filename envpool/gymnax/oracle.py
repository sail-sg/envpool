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
"""Test and documentation adapters for the pinned Gymnax source."""

from collections.abc import Callable
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from gymnax.environments.misc import meta_maze, point_robot
from gymnax.utils import load_mnist
from gymnax.visualize import vis_circle, vis_maze, vis_minatar
from gymnax.visualize.vis_gym import SUPPORTED_GYM_ENVS
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Classic control and toy text already provide these tasks under their own IDs.
_EXISTING = {*SUPPORTED_GYM_ENVS, "Catch-bsuite", "FrozenLake-misc"}
TASK_IDS = tuple(
    name for name in gymnax.registered_envs if name not in _EXISTING
)


def first_slot(info: dict) -> dict:
    """Select environment fields for a single-slot rollout."""
    return {k: v[0] for k, v in info.items() if k != "players"}


def make_oracle(name: str, **kwargs: Any) -> tuple[Any, Any]:
    """Construct the official task using Bazel's pinned data files."""
    if name != "MNISTBandit-bsuite":
        return gymnax.make(name, **kwargs)
    # Use the official IDX parser, including its signed-pixel conversion.
    loader = partial(load_mnist.load_mnist, directory=Path(__file__).parent)
    with patch.object(load_mnist, "load_mnist", loader):
        return gymnax.make(name, **kwargs)


def speeds(obs: np.ndarray) -> np.ndarray:
    """Recover each Freeway car's signed speed from its trail channel."""
    cars = obs[..., 1:9, :, 1].argmax(axis=-1)
    trails = obs[..., 1:9, :, 2:].reshape(*cars.shape, 50).argmax(axis=-1)
    direction = np.where((cars - trails // 5) % 10 == 1, 1, -1)
    return ((1 + trails % 5) * direction).astype(np.int32)


def synchronize(env: Any, p: Any, obs: np.ndarray, info: dict) -> Any:
    """Transfer only the randomized parts of a reset and check its observation."""
    name = env.name.split("-")[0]
    key = jax.random.key(0)
    h = info["hidden"]
    if name == "MNISTBandit":
        with patch.object(jax.random, "randint", return_value=jnp.int32(h[0])):
            image, state = env.reset_env(key, p)
        np.testing.assert_allclose(obs, image, atol=1e-7)
        return state

    _, s = env.reset(key, p)
    if name == "Breakout":
        x = np.int32(obs[3, :, 1].argmax())
        s = s.replace(ball_x=x, last_x=x, ball_dir=np.int32(2 if x == 0 else 3))
    elif name == "Freeway":
        v = speeds(obs)
        s = s.replace(cars=s.cars.at[:, 2].set(np.abs(v)).at[:, 3].set(v))
    elif name == "MemoryChain":
        s = s.replace(context=np.int32(obs[2:] > 0))
    elif name == "UmbrellaChain":
        s = s.replace(
            need_umbrella=np.int32(obs[0]), has_umbrella=np.int32(obs[1])
        )
    elif name == "SimpleBandit":
        s = s.replace(rewards=h)
    elif name == "BernoulliBandit":
        s = s.replace(reward_probs=jnp.array([h[0], 1 - h[0]]))
    elif name == "GaussianBandit":
        s = s.replace(mu=h[0])
    elif name == "MetaMaze":
        s = s.replace(goal=jnp.array(divmod(int(h[0]), 11)))
    elif name == "PointRobot":
        s = s.replace(pos=obs[:2], goal=h[:2])
    elif name == "Reacher":
        s = s.replace(
            angles=np.arctan2(obs[2:4], obs[:2]),
            angle_vels=obs[4:6],
            goal_xy=obs[6:],
        )
    elif name == "Swimmer":
        s = s.replace(
            urchin_xys=obs[:10].reshape(5, 2), xy=obs[10:12], goal_xy=obs[14:]
        )
    elif name == "Snake":
        head = np.argwhere(obs[:, :, 1])[0]
        apple = np.argwhere(obs[:, :, 2])[0]
        s = s.replace(
            head=np.int32(head),
            board=np.int32(obs[:, :, 1]),
            direction=np.int32(h[0]),
            apple=np.int32(apple),
        )
    s = jax.tree.map(jnp.asarray, s)
    expected = np.asarray(env.observe(key, s, None, p))
    assert expected.dtype == obs.dtype
    np.testing.assert_allclose(obs, expected, atol=3e-7, rtol=1e-6)
    return s


def stepper(env: Any, p: Any) -> Callable[..., Any]:
    """Condition the oracle on sampled events while it evolves its own state."""
    name = env.name.split("-")[0]
    key = jax.random.key(0)

    @jax.jit
    def step(s: Any, a: Any, h: Any) -> Any:
        # Replace only samplers during tracing. Gymnax still decides when
        # each sample is used and computes every transition and output.
        with ExitStack() as scope:

            def use(obj: Any, method: str, **value: Any) -> None:
                scope.enter_context(patch.object(obj, method, **value))

            if name == "Asterix":
                draws = h.astype(jnp.int32)
                use(jax.random, "permutation", return_value=draws[:8])
                use(jax.random, "choice", side_effect=[draws[8], draws[9]])
            elif name == "Seaquest":
                use(jax.random, "uniform", return_value=1 - h[:2])
                use(
                    jax.random,
                    "randint",
                    side_effect=[jnp.int32(h[2]), jnp.int32(h[4])],
                )
                use(jax.random, "bernoulli", return_value=h[3] != 0)
            elif name == "Freeway":
                use(jax.random, "randint", return_value=jnp.abs(h))
                use(jax.random, "choice", return_value=jnp.sign(h))
            elif name == "Snake":
                use(env, "_sample_apple", return_value=h)
            elif name == "MetaMaze":
                pos = jnp.int32(h[1])
                use(
                    meta_maze,
                    "reset_pos",
                    return_value=jnp.array([pos // 11, pos % 11]),
                )
            elif name == "PointRobot":
                use(point_robot, "sample_agent_position", return_value=h)
            elif name == "FourRooms":
                use(
                    jax.random,
                    "uniform",
                    return_value=jnp.where(h[0] < 0, 0.5, 0.0),
                )
                use(jax.random, "randint", return_value=jnp.int32(h[0]))
            elif name == "UmbrellaChain":
                use(
                    jax.random,
                    "bernoulli",
                    new=lambda key, p, shape: jnp.full(shape, h[0] != 0),
                )
            elif name == "BernoulliBandit":
                use(jax.random, "bernoulli", new=lambda key, p: h[1] < p)
            elif name == "GaussianBandit":
                use(jax.random, "normal", return_value=h[1])
            obs, s, reward, terminated, info = env.step_env(key, s, a, p)
            return obs, s, reward, terminated, env.is_truncated(s, p), info

    def compare_step(s: Any, a: Any, obs: np.ndarray, info: dict) -> Any:
        if name == "Freeway":
            sample = speeds(obs)
        elif name == "Snake":
            sample = np.argwhere(obs[:, :, 2])[0].astype(np.int32)
        elif name == "PointRobot":
            sample = obs[:2]
        else:
            sample = info["hidden"]
        return step(s, a, sample)

    return compare_step


_RENDERERS = {
    "FourRooms-misc": vis_maze.init_maze,
    "MetaMaze-misc": vis_maze.init_maze,
    "PointRobot-misc": vis_circle.init_circle,
    "Pong-misc": vis_minatar.init_minatar,
}


def renderable(name: str) -> bool:
    """Return whether Gymnax's Visualizer supports this task."""
    return name.endswith("-MinAtar") or name in _RENDERERS


def frame(env: Any, p: Any, state: Any, height: int, width: int) -> np.ndarray:
    """Render the scene without the Visualizer's score or animation margins."""
    figure = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    try:
        ax = figure.add_axes((0, 0, 1, 1))
        ax.set_axis_off()
        draw = _RENDERERS.get(env.name, vis_minatar.init_minatar)
        draw(ax, env, state, p)
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    finally:
        plt.close(figure)
