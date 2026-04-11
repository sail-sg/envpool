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
"""Oracle helpers for EnvPool gfootball tests."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from envpool.registration import base_path

from .gfootball_envpool import _GfootballOracleEngine

SMM_WIDTH = 96
SMM_HEIGHT = 72
_MINIMAP_NORM_X_MIN = -1.0
_MINIMAP_NORM_X_MAX = 1.0
_MINIMAP_NORM_Y_MIN = -1.0 / 2.25
_MINIMAP_NORM_Y_MAX = 1.0 / 2.25
_MARKER_VALUE = 255

_ACTION_IDLE = 0
_ACTION_LEFT = 1
_ACTION_RIGHT = 5
_ACTION_SHORT_PASS = 11
_ACTION_SET = np.asarray(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 20, 30, 14, 19, 31],
    dtype=np.int32,
)
_GAME_MODE_NORMAL = 0

TASK_SPECS: tuple[tuple[str, int], ...] = (
    ("11_vs_11_competition", 3000),
    ("11_vs_11_easy_stochastic", 3000),
    ("11_vs_11_hard_stochastic", 3000),
    ("11_vs_11_kaggle", 3000),
    ("11_vs_11_stochastic", 3000),
    ("1_vs_1_easy", 500),
    ("5_vs_5", 3000),
    ("academy_3_vs_1_with_keeper", 400),
    ("academy_corner", 400),
    ("academy_counterattack_easy", 400),
    ("academy_counterattack_hard", 400),
    ("academy_empty_goal", 400),
    ("academy_empty_goal_close", 400),
    ("academy_pass_and_shoot_with_keeper", 400),
    ("academy_run_pass_and_shoot_with_keeper", 400),
    ("academy_run_to_score", 400),
    ("academy_run_to_score_with_keeper", 400),
    ("academy_single_goal_versus_lazy", 3000),
)

TASK_CONFIG = {
    f"gfootball/{env_name}-v1": {
        "env_name": env_name,
        "max_episode_steps": max_episode_steps,
    }
    for env_name, max_episode_steps in TASK_SPECS
}

ALL_TASK_IDS = tuple(TASK_CONFIG)


def register_gfootball_envs() -> None:
    importlib.import_module("envpool.gfootball.registration")


def _mark_point(frame: np.ndarray, x: float, y: float) -> None:
    px = int(
        (x - _MINIMAP_NORM_X_MIN)
        / (_MINIMAP_NORM_X_MAX - _MINIMAP_NORM_X_MIN)
        * frame.shape[1]
    )
    py = int(
        (y - _MINIMAP_NORM_Y_MIN)
        / (_MINIMAP_NORM_Y_MAX - _MINIMAP_NORM_Y_MIN)
        * frame.shape[0]
    )
    px = max(0, min(frame.shape[1] - 1, px))
    py = max(0, min(frame.shape[0] - 1, py))
    frame[py, px] = _MARKER_VALUE


class GfootballOracle:
    """Small oracle that follows the upstream Python environment semantics."""

    def __init__(
        self,
        task_id: str,
        *,
        render: bool = False,
        render_resolution_x: int = 1280,
        render_resolution_y: int = 720,
        physics_steps_per_frame: int = 10,
    ) -> None:
        task = TASK_CONFIG[task_id]
        self._env_name = str(task["env_name"])
        self._max_episode_steps = int(task["max_episode_steps"])
        self._render = render
        self._render_resolution_x = int(render_resolution_x)
        self._render_resolution_y = int(render_resolution_y)
        self._engine = _GfootballOracleEngine(
            base_path,
            render,
            self._render_resolution_x,
            self._render_resolution_y,
            int(physics_steps_per_frame),
        )
        self._frame: np.ndarray | None = None
        self._observation: dict[str, Any] = {}
        self._previous_score_diff = 0
        self._previous_game_mode = -1
        self._prev_ball_owned_team = -1
        self._engine_seed = 0
        self._episode_number = 0
        self._step = 0
        self._elapsed_step = 0
        self._end_episode_on_score = False
        self._end_episode_on_possession_change = False
        self._end_episode_on_out_of_play = False

    def _retrieve_observation(self) -> bool:
        info = self._engine.get_info()
        observation: dict[str, Any] = {}
        if self._render:
            frame = np.frombuffer(self._engine.get_frame(), dtype=np.uint8)
            frame = np.reshape(
                frame,
                [
                    self._render_resolution_x,
                    self._render_resolution_y,
                    3,
                ],
            )
            frame = np.reshape(
                np.concatenate(
                    [frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]]
                ),
                [3, self._render_resolution_y, self._render_resolution_x],
            )
            frame = np.transpose(frame, [1, 2, 0])
            self._frame = np.flip(frame, 0)
        smm = np.zeros((SMM_HEIGHT, SMM_WIDTH, 4), dtype=np.uint8)
        for player in info["left_team"]:
            _mark_point(smm[:, :, 0], *player["position"])
        for player in info["right_team"]:
            _mark_point(smm[:, :, 1], *player["position"])
        _mark_point(smm[:, :, 2], info["ball_position"][0], info["ball_position"][1])
        active = (
            int(info["left_controllers"][0]) if info["left_controllers"] else -1
        )
        if 0 <= active < len(info["left_team"]):
            _mark_point(smm[:, :, 3], *info["left_team"][active]["position"])
        observation["obs"] = smm
        observation["active"] = np.int32(active)
        observation["score"] = np.asarray(
            [info["left_goals"], info["right_goals"]], dtype=np.int32
        )
        observation["game_mode"] = np.int32(info["game_mode"])
        observation["ball_owned_team"] = np.int32(info["ball_owned_team"])
        observation["ball_owned_player"] = np.int32(info["ball_owned_player"])
        observation["steps_left"] = np.int32(
            max(0, self._max_episode_steps - int(info["step"]))
        )
        self._observation = observation
        self._step = int(info["step"])
        return bool(info["is_in_play"])

    def reset(self, *, engine_seed: int, episode_number: int) -> tuple[np.ndarray, dict[str, Any]]:
        self._engine_seed = int(engine_seed)
        self._episode_number = int(episode_number)
        self._previous_score_diff = 0
        self._previous_game_mode = -1
        self._prev_ball_owned_team = -1
        self._elapsed_step = 0
        self._engine.waiting_for_game_count = 0
        self._engine.reset(
            self._env_name,
            self._episode_number,
            self._engine_seed,
            self._max_episode_steps,
        )
        while not self._retrieve_observation():
            self._engine.step()
        self._end_episode_on_score = self._env_name in {
            "academy_3_vs_1_with_keeper",
            "academy_corner",
            "academy_counterattack_easy",
            "academy_counterattack_hard",
            "academy_empty_goal",
            "academy_empty_goal_close",
            "academy_pass_and_shoot_with_keeper",
            "academy_run_pass_and_shoot_with_keeper",
            "academy_run_to_score",
            "academy_run_to_score_with_keeper",
            "academy_single_goal_versus_lazy",
        }
        self._end_episode_on_possession_change = self._env_name in {
            "academy_3_vs_1_with_keeper",
            "academy_counterattack_easy",
            "academy_counterattack_hard",
            "academy_empty_goal",
            "academy_empty_goal_close",
            "academy_pass_and_shoot_with_keeper",
            "academy_run_pass_and_shoot_with_keeper",
            "academy_run_to_score",
            "academy_run_to_score_with_keeper",
            "academy_single_goal_versus_lazy",
        }
        self._end_episode_on_out_of_play = self._env_name in {
            "academy_3_vs_1_with_keeper",
            "academy_corner",
            "academy_counterattack_easy",
            "academy_counterattack_hard",
            "academy_empty_goal",
            "academy_empty_goal_close",
            "academy_pass_and_shoot_with_keeper",
            "academy_run_pass_and_shoot_with_keeper",
            "academy_run_to_score",
            "academy_run_to_score_with_keeper",
            "academy_single_goal_versus_lazy",
        }
        return self.obs(), self.info()

    def obs(self) -> np.ndarray:
        return np.array(self._observation["obs"], copy=True)

    def info(self) -> dict[str, Any]:
        return {
            "score": np.array(self._observation["score"], copy=True),
            "game_mode": np.int32(self._observation["game_mode"]),
            "ball_owned_team": np.int32(self._observation["ball_owned_team"]),
            "ball_owned_player": np.int32(self._observation["ball_owned_player"]),
            "steps_left": np.int32(self._observation["steps_left"]),
            "engine_seed": np.int32(self._engine_seed),
            "episode_number": np.int32(self._episode_number),
            "elapsed_step": np.int32(self._elapsed_step),
        }

    def step(
        self, action: int
    ) -> tuple[np.ndarray, np.float32, np.bool_, np.bool_, dict[str, Any]]:
        backend_action = int(_ACTION_SET[int(action)])
        waiting = int(self._engine.waiting_for_game_count)
        if waiting == 20:
            backend_action = _ACTION_SHORT_PASS
        elif waiting > 20:
            backend_action = _ACTION_IDLE
            if (
                int(self._observation["ball_owned_team"]) == 0
                and int(self._observation["active"])
                == int(self._observation["ball_owned_player"])
            ):
                backend_action = _ACTION_RIGHT if waiting < 30 else _ACTION_LEFT
        self._engine.action(backend_action, True, 0)
        while True:
            self._engine.step()
            if self._retrieve_observation():
                break

        done = False
        if self._end_episode_on_score and np.any(self._observation["score"] > 0):
            done = True
        if (
            self._end_episode_on_out_of_play
            and int(self._observation["game_mode"]) != _GAME_MODE_NORMAL
            and self._previous_game_mode == _GAME_MODE_NORMAL
        ):
            done = True
        self._previous_game_mode = int(self._observation["game_mode"])
        if (
            self._end_episode_on_possession_change
            and int(self._observation["ball_owned_team"]) != -1
            and self._prev_ball_owned_team != -1
            and int(self._observation["ball_owned_team"])
            != self._prev_ball_owned_team
        ):
            done = True
        if int(self._observation["ball_owned_team"]) != -1:
            self._prev_ball_owned_team = int(self._observation["ball_owned_team"])

        score_diff = int(self._observation["score"][0] - self._observation["score"][1])
        reward = np.float32(score_diff - self._previous_score_diff)
        self._previous_score_diff = score_diff

        if int(self._observation["game_mode"]) != _GAME_MODE_NORMAL:
            self._engine.waiting_for_game_count = waiting + 1
        else:
            self._engine.waiting_for_game_count = 0
        if self._step >= self._max_episode_steps:
            done = True

        self._elapsed_step += 1

        truncated = np.bool_(done and self._step >= self._max_episode_steps)
        terminated = np.bool_(done and not truncated)
        return self.obs(), reward, terminated, truncated, self.info()

    def render(self) -> np.ndarray:
        assert self._frame is not None
        return np.ascontiguousarray(self._frame[:, :, ::-1])
