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
"""Registry, smoke, and determinism tests for MuJoCo Playground envs."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from absl.testing import absltest

import envpool.mujoco.playground.registration as playground_registration
from envpool.registration import list_all_envs, make_gymnasium, make_spec

_TASK_IDS = tuple(
    f"{name}-v1" for name in playground_registration.PLAYGROUND_ENVS
)
_DETERMINISM_STEPS = 32
_SEED_INDEPENDENT_TASK_IDS: frozenset[str] = frozenset()
_STATE_SHAPES = {
    "AlohaHandOver-v1": 83,
    "AlohaSinglePegInsertion-v1": 82,
    "ApolloJoystickFlatTerrain-v1": 112,
    "BarkourJoystick-v1": 465,
    "BerkeleyHumanoidJoystickFlatTerrain-v1": 52,
    "BerkeleyHumanoidJoystickRoughTerrain-v1": 52,
    "G1JoystickFlatTerrain-v1": 103,
    "G1JoystickRoughTerrain-v1": 103,
    "Go1JoystickFlatTerrain-v1": 48,
    "Go1JoystickRoughTerrain-v1": 48,
    "Go1Getup-v1": 42,
    "Go1Handstand-v1": 45,
    "Go1Footstand-v1": 45,
    "H1InplaceGaitTracking-v1": 186,
    "H1JoystickGaitTracking-v1": 113,
    "LeapCubeReorient-v1": 57,
    "LeapCubeRotateZAxis-v1": 32,
    "Op3Joystick-v1": 147,
    "PandaPickCube-v1": 66,
    "PandaPickCubeCartesian-v1": 70,
    "PandaPickCubeOrientation-v1": 66,
    "PandaOpenCabinet-v1": 55,
    "PandaRobotiqPushCube-v1": 48,
    "AeroCubeRotateZAxis-v1": 14,
    "SpotFlatTerrainJoystick-v1": 81,
    "SpotGetup-v1": 30,
    "SpotJoystickGaitTracking-v1": 69,
    "T1JoystickFlatTerrain-v1": 85,
    "T1JoystickRoughTerrain-v1": 85,
}
_ACTION_SHAPES = {
    "AlohaHandOver-v1": (14,),
    "AlohaSinglePegInsertion-v1": (14,),
    "ApolloJoystickFlatTerrain-v1": (32,),
    "G1JoystickFlatTerrain-v1": (29,),
    "G1JoystickRoughTerrain-v1": (29,),
    "H1InplaceGaitTracking-v1": (19,),
    "H1JoystickGaitTracking-v1": (19,),
    "LeapCubeReorient-v1": (16,),
    "LeapCubeRotateZAxis-v1": (16,),
    "Op3Joystick-v1": (20,),
    "PandaPickCube-v1": (8,),
    "PandaPickCubeCartesian-v1": (3,),
    "PandaPickCubeOrientation-v1": (8,),
    "PandaOpenCabinet-v1": (8,),
    "PandaRobotiqPushCube-v1": (7,),
    "AeroCubeRotateZAxis-v1": (7,),
    "T1JoystickFlatTerrain-v1": (23,),
    "T1JoystickRoughTerrain-v1": (23,),
}
_PRIVILEGED_STATE_SHAPES = {
    "ApolloJoystickFlatTerrain-v1": 224,
    "BerkeleyHumanoidJoystickFlatTerrain-v1": 114,
    "BerkeleyHumanoidJoystickRoughTerrain-v1": 114,
    "G1JoystickFlatTerrain-v1": 216,
    "G1JoystickRoughTerrain-v1": 216,
    "Go1JoystickFlatTerrain-v1": 123,
    "Go1JoystickRoughTerrain-v1": 123,
    "Go1Getup-v1": 91,
    "Go1Handstand-v1": 94,
    "Go1Footstand-v1": 94,
    "LeapCubeReorient-v1": 128,
    "LeapCubeRotateZAxis-v1": 105,
    "AeroCubeRotateZAxis-v1": 81,
    "SpotFlatTerrainJoystick-v1": 167,
    "T1JoystickFlatTerrain-v1": 180,
    "T1JoystickRoughTerrain-v1": 180,
}


def _zero_action(space: Any, num_envs: int) -> Any:
    if isinstance(space, gymnasium.spaces.Dict):
        return {
            key: _zero_action(subspace, num_envs)
            for key, subspace in space.spaces.items()
        }
    sample = np.asarray(space.sample())
    zero = np.zeros_like(sample)
    if sample.ndim == 0:
        return np.full((num_envs,), zero.item(), dtype=sample.dtype)
    return np.repeat(zero[np.newaxis, ...], num_envs, axis=0)


def _sample_actions(
    space: Any, rng: np.random.Generator, steps: int, num_envs: int
) -> list[Any]:
    if isinstance(space, gymnasium.spaces.Dict):
        return [
            {
                key: _sample_action(subspace, rng, num_envs)
                for key, subspace in space.spaces.items()
            }
            for _ in range(steps)
        ]
    return [_sample_action(space, rng, num_envs) for _ in range(steps)]


def _sample_action(space: Any, rng: np.random.Generator, num_envs: int) -> Any:
    if isinstance(space, gymnasium.spaces.Box):
        return rng.uniform(
            low=space.low,
            high=space.high,
            size=(num_envs, *space.shape),
        ).astype(space.dtype)
    return np.array([space.sample() for _ in range(num_envs)])


def _state_obs(obs: Any) -> np.ndarray:
    return np.asarray(obs["state"] if isinstance(obs, dict) else obs)


def _privileged_obs(obs: Any) -> np.ndarray | None:
    if not isinstance(obs, dict) or "privileged_state" not in obs:
        return None
    return np.asarray(obs["privileged_state"])


def _nested_arrays_differ(lhs: Any, rhs: Any) -> bool:
    if isinstance(lhs, dict):
        if not isinstance(rhs, dict) or lhs.keys() != rhs.keys():
            return True
        return any(_nested_arrays_differ(lhs[key], rhs[key]) for key in lhs)
    return not np.array_equal(np.asarray(lhs), np.asarray(rhs))


class PlaygroundTest(absltest.TestCase):
    """Validate registration, runtime surface, and determinism."""

    def _assert_nested_array_equal(
        self, lhs: Any, rhs: Any, label: str
    ) -> None:
        if isinstance(lhs, dict):
            self.assertIsInstance(rhs, dict, label)
            self.assertEqual(lhs.keys(), rhs.keys(), label)
            for key in lhs:
                self._assert_nested_array_equal(
                    lhs[key], rhs[key], f"{label}.{key}"
                )
            return
        np.testing.assert_array_equal(
            np.asarray(lhs), np.asarray(rhs), err_msg=label
        )

    def _assert_info_equal(
        self, info0: dict[str, Any], info1: dict[str, Any]
    ) -> None:
        self.assertEqual(info0.keys(), info1.keys())
        for key in info0:
            self._assert_nested_array_equal(
                info0[key], info1[key], f"info[{key}]"
            )

    def test_registered_tasks(self) -> None:
        """Checks public task IDs, alias resolution, and spaces."""
        registered = set(list_all_envs())
        self.assertEqual(
            _TASK_IDS,
            (
                "AlohaHandOver-v1",
                "AlohaSinglePegInsertion-v1",
                "ApolloJoystickFlatTerrain-v1",
                "BarkourJoystick-v1",
                "BerkeleyHumanoidJoystickFlatTerrain-v1",
                "BerkeleyHumanoidJoystickRoughTerrain-v1",
                "G1JoystickFlatTerrain-v1",
                "G1JoystickRoughTerrain-v1",
                "Go1JoystickFlatTerrain-v1",
                "Go1JoystickRoughTerrain-v1",
                "Go1Getup-v1",
                "Go1Handstand-v1",
                "Go1Footstand-v1",
                "H1InplaceGaitTracking-v1",
                "H1JoystickGaitTracking-v1",
                "LeapCubeReorient-v1",
                "LeapCubeRotateZAxis-v1",
                "Op3Joystick-v1",
                "PandaPickCube-v1",
                "PandaPickCubeCartesian-v1",
                "PandaPickCubeOrientation-v1",
                "PandaOpenCabinet-v1",
                "PandaRobotiqPushCube-v1",
                "AeroCubeRotateZAxis-v1",
                "SpotFlatTerrainJoystick-v1",
                "SpotGetup-v1",
                "SpotJoystickGaitTracking-v1",
                "T1JoystickFlatTerrain-v1",
                "T1JoystickRoughTerrain-v1",
            ),
        )
        for task_id in _TASK_IDS:
            with self.subTest(task_id=task_id):
                self.assertIn(task_id, registered)
                self.assertIn(f"MuJoCoPlayground/{task_id}", registered)
                spec = make_spec(task_id)
                alias_spec = make_spec(f"MuJoCoPlayground/{task_id}")
                if task_id in _PRIVILEGED_STATE_SHAPES:
                    self.assertIsInstance(
                        spec.gymnasium_observation_space,
                        gymnasium.spaces.Dict,
                    )
                    self.assertEqual(
                        spec.gymnasium_observation_space["state"].shape,
                        (_STATE_SHAPES[task_id],),
                    )
                    self.assertEqual(
                        spec.gymnasium_observation_space[
                            "privileged_state"
                        ].shape,
                        (_PRIVILEGED_STATE_SHAPES[task_id],),
                    )
                else:
                    self.assertIsInstance(
                        spec.gymnasium_observation_space,
                        gymnasium.spaces.Box,
                    )
                    self.assertEqual(
                        spec.gymnasium_observation_space.shape,
                        (_STATE_SHAPES[task_id],),
                    )
                self.assertEqual(
                    spec.action_space.shape,
                    _ACTION_SHAPES.get(task_id, (12,)),
                )
                self.assertEqual(
                    alias_spec.action_space.shape, spec.action_space.shape
                )

    def test_reset_and_step_reference_surface(self) -> None:
        """Checks reset and one control step expose the expected arrays."""
        for task_id in _TASK_IDS:
            with self.subTest(task_id=task_id):
                env = make_gymnasium(task_id, num_envs=2, seed=7)
                try:
                    obs, info = env.reset()
                    obs_state = _state_obs(obs)
                    self.assertEqual(
                        obs_state.shape, (2, _STATE_SHAPES[task_id])
                    )
                    obs_privileged = _privileged_obs(obs)
                    if task_id in _PRIVILEGED_STATE_SHAPES:
                        self.assertIsNotNone(obs_privileged)
                        assert obs_privileged is not None
                        self.assertEqual(
                            obs_privileged.shape,
                            (2, _PRIVILEGED_STATE_SHAPES[task_id]),
                        )
                    action = _zero_action(env.action_space, 2)
                    obs, rew, term, trunc, info = env.step(action)
                    obs_state = _state_obs(obs)
                    self.assertEqual(
                        obs_state.shape, (2, _STATE_SHAPES[task_id])
                    )
                    obs_privileged = _privileged_obs(obs)
                    if task_id in _PRIVILEGED_STATE_SHAPES:
                        self.assertIsNotNone(obs_privileged)
                        assert obs_privileged is not None
                        self.assertEqual(
                            obs_privileged.shape,
                            (2, _PRIVILEGED_STATE_SHAPES[task_id]),
                        )
                    self.assertEqual(rew.shape, (2,))
                    self.assertEqual(term.shape, (2,))
                    self.assertEqual(trunc.shape, (2,))
                    self.assertFalse(np.any(np.isnan(obs_state)))
                    if obs_privileged is not None:
                        self.assertFalse(np.any(np.isnan(obs_privileged)))
                    self.assertFalse(np.any(np.isnan(rew)))
                    if "Apollo" in task_id:
                        self.assertIn("command", info)
                        self.assertIn("steps_until_next_cmd", info)
                        self.assertIn("reward_tracking", info)
                    elif "Barkour" in task_id:
                        self.assertIn("command", info)
                        self.assertIn("reward_tracking_lin_vel", info)
                        self.assertIn("reward_feet_air_time", info)
                    elif "Op3" in task_id:
                        self.assertIn("command", info)
                        self.assertIn("reward_tracking_lin_vel", info)
                        self.assertIn("reward_zero_cmd", info)
                    elif "BerkeleyHumanoid" in task_id:
                        self.assertIn("command", info)
                        self.assertIn("reward_tracking_lin_vel", info)
                        self.assertIn("reward_feet_phase", info)
                    elif task_id.startswith("G1"):
                        self.assertIn("command", info)
                        self.assertIn("reward_tracking_lin_vel", info)
                        self.assertIn("reward_contact_force", info)
                    elif task_id.startswith("H1"):
                        self.assertIn("reward_feet_phase", info)
                        self.assertIn("reward_pose", info)
                        if "Joystick" in task_id:
                            self.assertIn("command", info)
                            self.assertIn("reward_tracking_lin_vel", info)
                        else:
                            self.assertIn("reward_ang_vel", info)
                    elif task_id.startswith(("Leap", "Aero")):
                        self.assertIn("reward_action_rate", info)
                        if task_id == "LeapCubeReorient-v1":
                            self.assertIn("reward_orientation", info)
                            self.assertIn("reward_position", info)
                            self.assertIn("reward_success", info)
                            self.assertIn("success_count", info)
                        else:
                            self.assertIn("reward_angvel", info)
                            self.assertIn("reward_termination", info)
                    elif task_id.startswith("Spot"):
                        if task_id == "SpotGetup-v1":
                            self.assertIn("reward_torso_height", info)
                            self.assertIn("reward_stand_still", info)
                        elif task_id == "SpotJoystickGaitTracking-v1":
                            self.assertIn("command", info)
                            self.assertIn("reward_feet_phase", info)
                            self.assertIn("reward_hip_splay", info)
                        else:
                            self.assertIn("command", info)
                            self.assertIn("reward_tracking_lin_vel", info)
                            self.assertIn("reward_feet_air_time", info)
                    elif task_id.startswith("T1"):
                        self.assertIn("command", info)
                        self.assertIn("reward_tracking_lin_vel", info)
                        self.assertIn("reward_feet_phase", info)
                        self.assertIn("reward_feet_distance", info)
                    elif task_id.startswith("Aloha"):
                        self.assertIn("reward_no_table_collision", info)
                        if task_id == "AlohaHandOver-v1":
                            self.assertIn("reward_gripper_box", info)
                            self.assertIn("reward_box_handover", info)
                            self.assertIn("reward_handover_target", info)
                        else:
                            self.assertIn("reward_left_reward", info)
                            self.assertIn("reward_peg_insertion_reward", info)
                            self.assertIn("peg_end2_dist_to_line", info)
                    elif task_id.startswith("Panda"):
                        self.assertIn("reward_gripper_box", info)
                        self.assertIn("reward_box_target", info)
                        self.assertIn("reward_robot_target_qpos", info)
                        if task_id == "PandaRobotiqPushCube-v1":
                            self.assertIn("reward_box_orientation", info)
                            self.assertIn("reward_joint_vel", info)
                            self.assertIn("reward_action_rate", info)
                            self.assertIn("success", info)
                        elif task_id == "PandaPickCubeCartesian-v1":
                            self.assertIn("reward_no_box_collision", info)
                            self.assertIn("reward_lifted", info)
                            self.assertIn("reward_success", info)
                        if task_id == "PandaOpenCabinet-v1":
                            self.assertIn("reward_no_barrier_collision", info)
                        elif task_id != "PandaRobotiqPushCube-v1":
                            self.assertIn("reward_no_floor_collision", info)
                    elif "Joystick" in task_id:
                        self.assertIn("command", info)
                        self.assertIn("steps_until_next_cmd", info)
                        self.assertIn("reward_tracking_lin_vel", info)
                    elif "Getup" in task_id:
                        self.assertIn("reward_torso_height", info)
                    elif "Handstand" in task_id or "Footstand" in task_id:
                        self.assertIn("reward_height", info)
                finally:
                    env.close()

    def test_reference_surface_is_bitwise_deterministic(self) -> None:
        """Checks same seed and action sequence reproduce bitwise rollouts."""
        rng = np.random.default_rng(123)
        for task_id in _TASK_IDS:
            with self.subTest(task_id=task_id):
                env0 = make_gymnasium(task_id, num_envs=2, seed=42)
                env1 = make_gymnasium(task_id, num_envs=2, seed=42)
                env2 = make_gymnasium(task_id, num_envs=2, seed=43)
                actions = _sample_actions(
                    env0.action_space, rng, _DETERMINISM_STEPS, 2
                )
                try:
                    obs0, info0 = env0.reset()
                    obs1, info1 = env1.reset()
                    obs2, _ = env2.reset()
                    self._assert_nested_array_equal(obs0, obs1, "reset_obs")
                    self._assert_info_equal(info0, info1)
                    differs = _nested_arrays_differ(obs0, obs2)
                    for action in actions:
                        step0 = env0.step(action)
                        step1 = env1.step(action)
                        step2 = env2.step(action)
                        for index, (value0, value1) in enumerate(
                            zip(step0[:4], step1[:4], strict=True)
                        ):
                            self._assert_nested_array_equal(
                                value0, value1, f"step[{index}]"
                            )
                        self._assert_info_equal(step0[4], step1[4])
                        differs = differs or _nested_arrays_differ(
                            step0[0], step2[0]
                        )
                    if task_id not in _SEED_INDEPENDENT_TASK_IDS:
                        self.assertTrue(differs, task_id)
                finally:
                    env0.close()
                    env1.close()
                    env2.close()


if __name__ == "__main__":
    absltest.main()
