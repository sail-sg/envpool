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
"""Independent alignment tests against the pinned official Jumanji oracle."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

# Configure the test-only backend before importing JAX on Windows.
from envpool.jumanji.oracle import (
    _as_numpy,
    _envpool_kwargs_from_official_state,
    _find_viewer,
    _make_official_env,
    _minesweeper_flat_mines,
    _official_render,
)

# isort: split

import jax  # noqa: E402
import numpy as np  # noqa: E402
from absl.testing import absltest, parameterized  # noqa: E402

import envpool.jumanji.registration as jumanji_registration  # noqa: E402
from envpool.registration import make_gymnasium  # noqa: E402

_RENDER_ALIGN_STEPS = 3
_RENDER_SIZE = 128


def _tree_to_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _tree_to_dict(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {
            key: _tree_to_dict(getattr(value, key)) for key in value._fields
        }
    if is_dataclass(value):
        return {
            field.name: _tree_to_dict(getattr(value, field.name))
            for field in fields(value)
        }
    return _as_numpy(value)


def _first_env_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _first_env_tree(item) for key, item in value.items()}
    array = np.asarray(value)
    assert array.shape[:1] == (1,), array.shape
    return array[0]


def _project(expected: Any, actual: Any) -> Any:
    if isinstance(actual, dict):
        assert isinstance(expected, dict), (expected, actual)
        return {key: _project(expected[key], actual[key]) for key in actual}
    return expected


def _assert_tree_values(actual: Any, expected: Any, label: str) -> None:
    if isinstance(actual, dict):
        assert isinstance(expected, dict), label
        assert actual.keys() == expected.keys(), label
        for key in actual:
            if key == "ems" and "ems_mask" in actual:
                # Inactive EMS slots are scratch storage in upstream; only the
                # mask and active coordinates/order participate in the task.
                mask = np.asarray(expected["ems_mask"], dtype=bool)
                for axis in actual[key]:
                    assert actual[key][axis].shape == expected[key][axis].shape
                    _assert_tree_values(
                        actual[key][axis][mask],
                        expected[key][axis][mask],
                        f"{label}.ems.{axis}",
                    )
            else:
                _assert_tree_values(
                    actual[key], expected[key], f"{label}.{key}"
                )
        return
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    assert actual_array.shape == expected_array.shape, label
    if np.issubdtype(expected_array.dtype, np.floating):
        if label.startswith("Snake-v1 ") and label.endswith("obs.grid"):
            # Body/head/tail/fruit channels encode discrete occupancy.
            np.testing.assert_array_equal(
                actual_array[..., :4], expected_array[..., :4], err_msg=label
            )
        atol = 1e-6
        if label.startswith("SearchAndRescue-v0 "):
            # Float32 integration plus ordinary inverse-normal math. Across
            # three 400-step macOS/Linux rollouts: positions <= 3.61e-6 and
            # normalized view distances <= 2.93e-5; visibility stayed exact.
            if label.endswith("positions"):
                atol = 5e-6
            elif label.endswith("searcher_views"):
                np.testing.assert_array_equal(
                    actual_array == -1, expected_array == -1, err_msg=label
                )
                atol = 5e-5
        np.testing.assert_allclose(
            actual_array, expected_array, rtol=1e-6, atol=atol, err_msg=label
        )
    else:
        np.testing.assert_array_equal(
            actual_array, expected_array, err_msg=label
        )
    # Jumanji 1.1.2 declares SearchAndRescue.step as int32, emits int64 on
    # reset, and int32 after step. EnvPool keeps the public spec dtype stable.
    if label.startswith("SearchAndRescue-v0 ") and label.endswith("obs.step"):
        return
    if label.startswith("PacMan-v1 ") and (
        label.endswith(("obs.player_locations.x", "obs.player_locations.y"))
    ):
        return
    assert actual_array.dtype == expected_array.dtype, (
        f"{label} dtype mismatch: {actual_array.dtype} != {expected_array.dtype}"
    )


def _envpool_action(action: Any) -> np.ndarray:
    action_array = _as_numpy(action)
    return action_array.reshape((1, *action_array.shape))


def _assert_reward_values(
    actual: Any,
    expected: Any,
    info: dict[str, Any],
    task_id: str,
    label: str,
) -> None:
    expected = _as_numpy(expected)
    if task_id == "Connector-v3":
        np.testing.assert_allclose(
            info["agent_rewards"][0],
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg=label,
        )
        expected = np.float32(expected.mean(dtype=np.float64))
    if task_id == "SearchAndRescue-v0":
        np.testing.assert_allclose(
            info["searcher_rewards"][0],
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg=label,
        )
        expected = expected.sum(dtype=np.float32)
    np.testing.assert_allclose(
        np.asarray(actual)[0], expected, rtol=1e-6, atol=1e-6, err_msg=label
    )


def _first_valid_action(
    official_env: Any,
    observation: Any,
    task_id: str | None = None,
    official_state: Any | None = None,
    rng: np.random.Generator | None = None,
) -> Any:
    spec = official_env.action_spec
    action = _as_numpy(spec.generate_value())
    choose = lambda values: values[0] if rng is None else rng.choice(values)
    if rng is not None:
        if np.issubdtype(action.dtype, np.floating):
            action = rng.uniform(
                spec.minimum, spec.maximum, size=action.shape
            ).astype(action.dtype)
        else:
            action = rng.integers(
                spec.minimum,
                spec.maximum + 1,
                size=action.shape,
                dtype=action.dtype,
            )
    if task_id == "Minesweeper-v0" and official_state is not None:
        board = _as_numpy(observation.board).reshape(-1)
        mines = _minesweeper_flat_mines(official_state)
        safe = [
            location
            for location, value in enumerate(board)
            if int(value) == -1 and location not in mines
        ]
        if safe:
            location = choose(safe)
            return np.asarray(
                [location // 10, location % 10], dtype=action.dtype
            )
    if np.issubdtype(action.dtype, np.floating):
        return np.zeros_like(action) if rng is None else action
    if not hasattr(observation, "action_mask"):
        return action

    mask = _as_numpy(observation.action_mask).astype(bool, copy=False)
    if not np.any(mask):
        return action

    if action.shape == () and mask.ndim == 1:
        return np.asarray(choose(np.flatnonzero(mask)), dtype=action.dtype)

    if action.shape == mask.shape[:-1]:
        selected = np.zeros_like(action)
        for prefix in np.ndindex(mask.shape[:-1]):
            valid = np.flatnonzero(mask[prefix])
            selected[prefix] = choose(valid) if valid.size else action[prefix]
        return selected

    if action.ndim == 1 and action.shape[0] == mask.ndim:
        return np.asarray(
            np.argwhere(mask)[choose(np.arange(np.count_nonzero(mask)))],
            dtype=action.dtype,
        )

    return action


class JumanjiOfficialAlignTest(parameterized.TestCase):
    """Official oracle alignment for every registered Jumanji task."""

    def test_native_rollout_and_render_surface(self) -> None:
        """Check reset plus three official-driven steps for every task."""
        for task_id in jumanji_registration.jumanji_env_ids:
            with self.subTest(task_id=task_id):
                official_env = _make_official_env(task_id)
                official_state, official_timestep = official_env.reset(
                    jax.random.PRNGKey(0)
                )
                envpool_kwargs = _envpool_kwargs_from_official_state(
                    task_id, official_state, official_timestep.observation
                )
                if task_id == "Sokoban-v0":
                    envpool_kwargs.update(
                        base_path="/tmp/envpool-missing-boxoban",
                        sokoban_level_index=0,
                    )
                env = make_gymnasium(
                    task_id,
                    num_envs=1,
                    seed=0,
                    render_mode="rgb_array",
                    render_width=_RENDER_SIZE,
                    render_height=_RENDER_SIZE,
                    **envpool_kwargs,
                )
                official_step = jax.jit(official_env.step)
                viewer = _find_viewer(official_env)
                try:
                    envpool_obs, envpool_info = env.reset()
                    expected_obs = _project(
                        _tree_to_dict(official_timestep.observation),
                        _first_env_tree(envpool_obs),
                    )
                    _assert_tree_values(
                        _first_env_tree(envpool_obs),
                        expected_obs,
                        f"{task_id} reset obs",
                    )
                    self.assertEqual(envpool_info["env_id"].tolist(), [0])
                    self._assert_render_surface(
                        env, official_env, official_state, task_id, 0
                    )

                    for step in range(1, _RENDER_ALIGN_STEPS + 1):
                        official_action = _first_valid_action(
                            official_env,
                            official_timestep.observation,
                            task_id=task_id,
                            official_state=official_state,
                        )
                        official_state, official_timestep = official_step(
                            official_state, official_action
                        )
                        (
                            envpool_obs,
                            envpool_reward,
                            envpool_terminated,
                            envpool_truncated,
                            envpool_info,
                        ) = env.step(_envpool_action(official_action))
                        expected_obs = _project(
                            _tree_to_dict(official_timestep.observation),
                            _first_env_tree(envpool_obs),
                        )
                        _assert_tree_values(
                            _first_env_tree(envpool_obs),
                            expected_obs,
                            f"{task_id} step {step} obs",
                        )
                        _assert_reward_values(
                            envpool_reward,
                            official_timestep.reward,
                            envpool_info,
                            task_id,
                            f"{task_id} step {step} reward",
                        )
                        np.testing.assert_array_equal(
                            np.asarray(envpool_terminated)[0],
                            bool(official_timestep.last()),
                            err_msg=f"{task_id} step {step} terminated",
                        )
                        np.testing.assert_array_equal(
                            np.asarray(envpool_truncated)[0],
                            False,
                            err_msg=f"{task_id} step {step} truncated",
                        )
                        self.assertEqual(envpool_info["env_id"].tolist(), [0])
                        self._assert_render_surface(
                            env, official_env, official_state, task_id, step
                        )
                finally:
                    env.close()
                    viewer.close()

    @parameterized.named_parameters(("seed_0", 0), ("seed_7", 7))
    def test_independent_rollouts(self, seed: int) -> None:
        """Compare varied actions through a complete episode of every task."""
        max_steps_by_task_id = {
            task_id: max_episode_steps
            for task_id, _, max_episode_steps in jumanji_registration._TASKS
        }
        for task_id in jumanji_registration.jumanji_env_ids:
            with self.subTest(task_id=task_id):
                rng = np.random.default_rng(seed + 41)
                official_env = _make_official_env(task_id)
                official_state, official_timestep = official_env.reset(
                    jax.random.PRNGKey(seed)
                )
                envpool_kwargs = _envpool_kwargs_from_official_state(
                    task_id, official_state, official_timestep.observation
                )
                if task_id == "Sokoban-v0":
                    envpool_kwargs.update(
                        base_path="/tmp/envpool-missing-boxoban",
                        sokoban_level_index=0,
                    )
                env = make_gymnasium(
                    task_id,
                    num_envs=1,
                    seed=seed,
                    **envpool_kwargs,
                )
                official_step = jax.jit(official_env.step)
                try:
                    envpool_obs, _ = env.reset()
                    expected_obs = _project(
                        _tree_to_dict(official_timestep.observation),
                        _first_env_tree(envpool_obs),
                    )
                    _assert_tree_values(
                        _first_env_tree(envpool_obs),
                        expected_obs,
                        f"{task_id} long reset obs",
                    )
                    rollout_steps = max_steps_by_task_id[task_id]
                    done_step = None
                    for step in range(1, rollout_steps + 1):
                        official_action = _first_valid_action(
                            official_env,
                            official_timestep.observation,
                            task_id=task_id,
                            official_state=official_state,
                            rng=rng,
                        )
                        official_state, official_timestep = official_step(
                            official_state, official_action
                        )
                        (
                            envpool_obs,
                            envpool_reward,
                            envpool_terminated,
                            envpool_truncated,
                            envpool_info,
                        ) = env.step(_envpool_action(official_action))
                        expected_obs = _project(
                            _tree_to_dict(official_timestep.observation),
                            _first_env_tree(envpool_obs),
                        )
                        _assert_tree_values(
                            _first_env_tree(envpool_obs),
                            expected_obs,
                            f"{task_id} long step {step} obs",
                        )
                        _assert_reward_values(
                            envpool_reward,
                            official_timestep.reward,
                            envpool_info,
                            task_id,
                            f"{task_id} long step {step} reward",
                        )
                        official_done = bool(
                            _as_numpy(official_timestep.last())
                        )
                        official_terminated = official_done and not np.any(
                            _as_numpy(official_timestep.discount)
                        )
                        envpool_done = bool(
                            envpool_terminated[0] or envpool_truncated[0]
                        )
                        self.assertEqual(
                            bool(envpool_terminated[0]),
                            official_terminated,
                            msg=f"{task_id} long step {step} terminated",
                        )
                        self.assertEqual(
                            bool(envpool_truncated[0]),
                            official_done and not official_terminated,
                            msg=f"{task_id} long step {step} truncated",
                        )
                        self.assertEqual(
                            envpool_info["env_id"].tolist(),
                            [0],
                            msg=f"{task_id} long step {step} env_id",
                        )
                        if envpool_done or official_done:
                            done_step = step
                            break
                    if (
                        done_step is None
                        and rollout_steps == max_steps_by_task_id[task_id]
                    ):
                        self.fail(f"{task_id} did not finish within max steps")
                finally:
                    env.close()

    def _assert_render_surface(
        self,
        env: Any,
        official_env: Any,
        official_state: Any,
        task_id: str,
        step: int,
    ) -> None:
        actual = np.asarray(env.render(env_ids=[0])[0], dtype=np.uint8)
        expected = _official_render(official_env, official_state)
        self.assertEqual(actual.shape, expected.shape)
        # Native and Matplotlib views intentionally use different drawing
        # styles. State agreement is checked above; separate regression tests
        # cover scene content and asynchronous frame selection.
        self.assertEqual(actual.dtype, np.uint8)
        self.assertGreater(int(np.ptp(actual)), 0)
        self.assertGreater(int(np.ptp(expected)), 0)


if __name__ == "__main__":
    absltest.main()
