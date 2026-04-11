#!/usr/bin/env python3
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

"""Generate a C++ scenario config include from upstream gfootball scenarios."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SCENARIOS = (
    "11_vs_11_competition",
    "11_vs_11_easy_stochastic",
    "11_vs_11_hard_stochastic",
    "11_vs_11_kaggle",
    "11_vs_11_stochastic",
    "1_vs_1_easy",
    "5_vs_5",
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
)

ROLE_NAMES = (
    "e_PlayerRole_GK",
    "e_PlayerRole_CB",
    "e_PlayerRole_LB",
    "e_PlayerRole_RB",
    "e_PlayerRole_DM",
    "e_PlayerRole_CM",
    "e_PlayerRole_LM",
    "e_PlayerRole_RM",
    "e_PlayerRole_AM",
    "e_PlayerRole_CF",
)


@dataclass(eq=True)
class ScenarioState:
    ball_position: tuple[float, float] = (0.0, 0.0)
    left_team: list[tuple[float, float, str, bool, bool]] = field(
        default_factory=list
    )
    right_team: list[tuple[float, float, str, bool, bool]] = field(
        default_factory=list
    )
    left_agents: int = 1
    right_agents: int = 0
    use_magnet: bool = True
    offsides: bool = True
    real_time: bool = False
    left_team_difficulty: float = 1.0
    right_team_difficulty: float = 0.6
    deterministic: bool = False
    end_episode_on_score: bool = False
    end_episode_on_possession_change: bool = False
    end_episode_on_out_of_play: bool = False
    game_duration: int = 3000
    control_all_players: bool = False
    second_half: int = 999999999


class Team:
    e_Left = "e_Left"
    e_Right = "e_Right"


class ScenarioBuilderCapture:
    def __init__(self, episode_number: int) -> None:
        self._episode_number = episode_number
        self._active_team = Team.e_Left
        self._config = ScenarioState()

    def config(self) -> ScenarioState:
        return self._config

    def SetTeam(self, team: str) -> None:
        self._active_team = team

    def AddPlayer(
        self,
        x: float,
        y: float,
        role: str,
        lazy: bool = False,
        controllable: bool = True,
    ) -> None:
        player = (float(x), float(y), str(role), bool(lazy), bool(controllable))
        if self._active_team == Team.e_Left:
            self._config.left_team.append(player)
        else:
            self._config.right_team.append(player)

    def SetBallPosition(self, ball_x: float, ball_y: float) -> None:
        self._config.ball_position = (float(ball_x), float(ball_y))

    def EpisodeNumber(self) -> int:
        return self._episode_number


def _install_fake_gfootball_modules(scenarios_dir: Path) -> None:
    fake_engine = types.ModuleType("gfootball_engine")
    fake_engine.e_PlayerRole = types.SimpleNamespace(
        **{name: name for name in ROLE_NAMES}
    )
    fake_engine.e_Team = Team
    sys.modules["gfootball_engine"] = fake_engine

    gfootball_pkg = types.ModuleType("gfootball")
    gfootball_pkg.__path__ = [str(scenarios_dir.parent)]
    sys.modules["gfootball"] = gfootball_pkg

    init_path = scenarios_dir / "__init__.py"
    init_spec = importlib.util.spec_from_file_location(
        "gfootball.scenarios",
        init_path,
        submodule_search_locations=[str(scenarios_dir)],
    )
    if init_spec is None or init_spec.loader is None:
        raise RuntimeError(f"Failed to load scenario package from {init_path}")
    scenarios_pkg = importlib.util.module_from_spec(init_spec)
    sys.modules["gfootball.scenarios"] = scenarios_pkg
    init_spec.loader.exec_module(scenarios_pkg)
    gfootball_pkg.scenarios = scenarios_pkg


def _load_scenario_module(path: Path):
    module_name = f"gfootball.scenarios.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load scenario module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _capture_state(path: Path, episode_number: int) -> ScenarioState:
    module = _load_scenario_module(path)
    builder = ScenarioBuilderCapture(episode_number)
    module.build_scenario(builder)
    return builder.config()


def _cpp_bool(value: bool) -> str:
    return "true" if value else "false"


def _cpp_float(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        text = "0"
    if "." not in text:
        text += ".0"
    return f"{text}f"


def _emit_state(state: ScenarioState, indent: str) -> list[str]:
    lines = [
        f"{indent}cfg->left_agents = {state.left_agents};",
        f"{indent}cfg->right_agents = {state.right_agents};",
        f"{indent}cfg->use_magnet = {_cpp_bool(state.use_magnet)};",
        (
            f"{indent}cfg->ball_position = Vector3("
            f"{_cpp_float(state.ball_position[0])}, "
            f"{_cpp_float(state.ball_position[1])}, 0.0f);"
        ),
        f"{indent}cfg->offsides = {_cpp_bool(state.offsides)};",
        f"{indent}cfg->real_time = {_cpp_bool(state.real_time)};",
        f"{indent}cfg->left_team_difficulty = {_cpp_float(state.left_team_difficulty)};",
        f"{indent}cfg->right_team_difficulty = {_cpp_float(state.right_team_difficulty)};",
        f"{indent}cfg->deterministic = {_cpp_bool(state.deterministic)};",
        (
            f"{indent}cfg->end_episode_on_score = "
            f"{_cpp_bool(state.end_episode_on_score)};"
        ),
        (
            f"{indent}cfg->end_episode_on_possession_change = "
            f"{_cpp_bool(state.end_episode_on_possession_change)};"
        ),
        (
            f"{indent}cfg->end_episode_on_out_of_play = "
            f"{_cpp_bool(state.end_episode_on_out_of_play)};"
        ),
        f"{indent}cfg->game_duration = {state.game_duration};",
        f"{indent}cfg->control_all_players = {_cpp_bool(state.control_all_players)};",
        f"{indent}cfg->second_half = {state.second_half};",
        f"{indent}cfg->left_team.clear();",
    ]
    for x, y, role, lazy, controllable in state.left_team:
        lines.append(
            f"{indent}cfg->left_team.emplace_back("
            f"{_cpp_float(x)}, {_cpp_float(y)}, {role}, "
            f"{_cpp_bool(lazy)}, {_cpp_bool(controllable)});"
        )
    lines.append(f"{indent}cfg->right_team.clear();")
    for x, y, role, lazy, controllable in state.right_team:
        lines.append(
            f"{indent}cfg->right_team.emplace_back("
            f"{_cpp_float(x)}, {_cpp_float(y)}, {role}, "
            f"{_cpp_bool(lazy)}, {_cpp_bool(controllable)});"
        )
    return lines


def _generate(paths: dict[str, Path]) -> str:
    scenarios_dir = next(iter(paths.values())).parent
    _install_fake_gfootball_modules(scenarios_dir)

    lines = [
        "inline void BuildScenarioConfig(const std::string& env_name,",
        "                                int episode_number,",
        "                                ScenarioConfig* cfg) {",
        "  const bool even_episode = (episode_number % 2) == 0;",
    ]
    for name in SUPPORTED_SCENARIOS:
        path = paths.get(name)
        if path is None:
            raise RuntimeError(f"Missing upstream scenario source for {name}")
        even_state = _capture_state(path, episode_number=2)
        odd_state = _capture_state(path, episode_number=1)
        lines.append(f"  if (env_name == \"{name}\") {{")
        if even_state == odd_state:
            lines.extend(_emit_state(even_state, "    "))
        else:
            lines.append("    if (even_episode) {")
            lines.extend(_emit_state(even_state, "      "))
            lines.append("    } else {")
            lines.extend(_emit_state(odd_state, "      "))
            lines.append("    }")
        lines.append("    return;")
        lines.append("  }")
    lines.append(
        "  throw std::runtime_error(\"Unknown gfootball scenario: \" + env_name);"
    )
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    paths: dict[str, Path] = {}
    for arg in argv[1:]:
        path = Path(arg)
        if path.suffix != ".py":
            continue
        if path.stem not in SUPPORTED_SCENARIOS:
            continue
        paths[path.stem] = path.resolve()
    if not paths:
        raise RuntimeError("No upstream scenario sources were provided")
    sys.stdout.write(_generate(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
