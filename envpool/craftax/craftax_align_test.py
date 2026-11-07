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
"""Step-level oracle acceptance for every public game and reset mode."""

from typing import Any

import numpy as np
from absl.testing import absltest, parameterized

import envpool.craftax.registration  # noqa: F401
from envpool.craftax.oracle import (
    assert_floats,
    assert_render,
    assert_state,
    encode,
    factory_names,
    jax,
    make_oracle,
    reset_info,
    rollout_action,
    state_layout,
    symbolic_observer,
    sync_world,
)
from envpool.registration import make_gymnasium


class CraftaxAlignmentTest(parameterized.TestCase):
    """Validate every public name against complete official rollouts."""

    @parameterized.parameters(*factory_names())
    def test_complete_rollouts(self, task_id: str) -> None:
        """Compare observations, rewards, state, information, and rendering across resets."""
        self.rollout(task_id)

    @parameterized.parameters(*factory_names())
    def test_initial_state_and_human_render(self, task_id: str) -> None:
        """Inject once, preserve terminal achievements, and render at human resolution."""
        self.rollout(
            task_id, seed=17, max_steps=64, steps=145, initial=True, tile=64
        )

    @parameterized.parameters("Craftax-Symbolic-v1", "Craftax-Pixels-v1")
    def test_god_mode_outside_world(self, task_id: str) -> None:
        """Walk outside the map and check padded camera clipping."""
        self.rollout(task_id, max_steps=192, steps=192, god_mode=True)

    @parameterized.parameters((True, 16), (True, 32), (False, 64))
    def test_configured_worlds(self, classic: bool, size: int) -> None:
        """Exercise map sizes, noise overrides, mob capacities, and game parameters."""
        task_id = "Craftax-" + ("Classic-" if classic else "") + "Symbolic-v1"
        oracle, params = make_oracle(task_id, 137)
        capacities = dict(
            max_melee_mobs=4,
            max_passive_mobs=2,
            max_ranged_mobs=3,
            max_mob_projectiles=5,
            max_growing_plants=6,
        )
        names = (
            dict(
                max_melee_mobs="max_zombies",
                max_passive_mobs="max_cows",
                max_ranged_mobs="max_skeletons",
                max_mob_projectiles="max_arrows",
            )
            if classic
            else {}
        )
        static = oracle.static_env_params.replace(
            map_size=(size, size),
            **{
                names.get(name, name): value
                for name, value in capacities.items()
            },
        )
        oracle = type(oracle)(static)
        shapes = ((size // 16 + 1, size // 16 + 1),) * 2 + (
            (size // 8 + 1, size // 2 + 1),
            (size // 4 + 1, size // 4 + 1),
        )
        rng = np.random.default_rng(17)
        angles = tuple(rng.random(shape, dtype=np.float32) for shape in shapes)
        dynamic = dict(day_length=96, always_diamond=True)
        if classic:
            dynamic.update(zombie_health=6, cow_health=4, skeleton_health=5)
        else:
            dynamic.update(max_attribute=7)
        params = params.replace(fractal_noise_angles=angles, **dynamic)
        options = dict(
            map_size=[size, size],
            fractal_noise_angles=[
                angle.reshape(-1).tolist() for angle in angles
            ],
            **capacities,
            **dynamic,
        )
        self.rollout(
            task_id,
            seed=23,
            max_steps=137,
            steps=420,
            reference=(oracle, params),
            options=options,
        )

    def rollout(
        self,
        task_id: str,
        *,
        seed: int = 1,
        max_steps: int = 257,
        steps: int = 800,
        initial: bool = False,
        tile: int = 16,
        god_mode: bool = False,
        reference: tuple[Any, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Copy generated terrain at reset and compare independent episodes."""
        classic = "-Classic-" in task_id
        pixel = "-Pixels-" in task_id
        auto = "-AutoReset-" in task_id
        oracle, params = reference or make_oracle(task_id, max_steps)
        # The native suite exercises default spawns. The short despawn range
        # leaves no eligible surface spawn positions in this continuous rollout.
        params = params.replace(mob_despawn_distance=1)
        if god_mode:
            params = params.replace(god_mode=True)

        @jax.jit
        def step(key: Any, state: Any, action: int, params: Any) -> Any:
            return oracle.step_env(key, state, action, params)[1:]

        get_obs = symbolic_observer(oracle, classic)

        def assert_observation(
            obs: np.ndarray, state: Any, message: str = ""
        ) -> None:
            if pixel:
                assert_render(
                    obs * 255, state, classic, 7 if classic else 10, message
                )
            else:
                assert_floats(obs, get_obs(state), message)

        key = jax.random.PRNGKey(seed + 1000)
        _, reset_state = oracle.reset(key, params)
        layout = state_layout(reset_state, classic)
        state = reset_state
        kwargs: dict[str, Any] = dict(options or {})
        if initial:
            state = state.replace(
                inventory=state.inventory.replace(
                    wood=jax.numpy.asarray(7, dtype=jax.numpy.int32)
                ),
                achievements=jax.numpy.ones_like(state.achievements),
            )
            kwargs["initial_state"] = encode(state, layout).tolist()
        env = make_gymnasium(
            task_id,
            num_envs=1,
            seed=seed,
            max_episode_steps=max_steps,
            debug_state=True,
            render_mode="rgb_array",
            render_tile_size=tile,
            god_mode=god_mode,
            mob_despawn_distance=1,
            **kwargs,
        )
        obs, info = env.reset()
        space = oracle.observation_space(params)
        self.assertEqual(env.observation_space.shape, space.shape)
        self.assertEqual(env.observation_space.dtype, space.dtype)
        self.assertEqual(env.action_space.n, oracle.num_actions)
        if not initial:
            state = sync_world(
                reset_state, info["state"][0], layout, classic, params
            )
        assert_observation(obs[0], state)
        assert_state(info["state"][0], state, layout)
        assert_render(np.asarray(env.render())[0], state, classic, tile)
        rng = np.random.default_rng(2026)
        done = False
        episodes = 0
        try:
            for t in range(steps):
                key, draw = jax.random.split(key)
                action = int(rng.integers(env.action_space.n))
                if god_mode:
                    action = 1 if t < 96 else 2
                else:
                    action = rollout_action(state, action, classic)
                reset = done and not auto
                obs, reward, terminated, truncated, info = env.step(
                    np.array([action], np.int32)
                )
                expected_trunc = False
                if reset:
                    state = sync_world(
                        reset_state, info["state"][0], layout, classic, params
                    )
                    expected_reward, expected_done = 0.0, False
                    expected_info = reset_info(state, classic)
                else:
                    state, expected_reward, expected_done, expected_info = step(
                        draw, state, action, params
                    )
                    if bool(expected_done):
                        expected_trunc = not bool(
                            oracle.is_terminal(
                                state,
                                params.replace(max_timesteps=max_steps + 1),
                            )
                        )
                message = f"{task_id} step {t}, action {action}"
                np.testing.assert_array_equal(
                    reward[0], expected_reward, err_msg=message
                )
                self.assertEqual(
                    bool(terminated[0]),
                    bool(expected_done) and not expected_trunc,
                    message,
                )
                self.assertEqual(bool(truncated[0]), expected_trunc, message)
                # The official terminal reward, flags and info refer to the
                # old episode even when the pool publishes a fresh observation.
                if bool(expected_done) and auto:
                    state = sync_world(
                        reset_state, info["state"][0], layout, classic, params
                    )
                assert_state(info["state"][0], state, layout, message)
                assert_observation(obs[0], state, message)
                for name, value in expected_info.items():
                    if name == "score":
                        assert_floats(info[name][0], value, message + " score")
                    else:
                        np.testing.assert_array_equal(
                            info[name][0], value, err_msg=message + " " + name
                        )
                if t % 29 == 0 or bool(expected_done):
                    assert_render(
                        np.asarray(env.render())[0],
                        state,
                        classic,
                        tile,
                        message,
                    )
                done = bool(expected_done)
                episodes += done
            self.assertGreaterEqual(episodes, steps // max_steps)
            if initial:
                # A later explicit reset must not reuse the injected inventory
                # or achievements.
                obs, info = env.reset()
                state = sync_world(
                    reset_state, info["state"][0], layout, classic, params
                )
                assert_observation(obs[0], state)
                assert_state(info["state"][0], state, layout)
        finally:
            env.close()


if __name__ == "__main__":
    absltest.main()
