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
"""PGX env registration."""

from envpool.registration import register

_COMMON = {
    "import_path": "envpool.pgx",
    "spec_cls": "GoEnvSpec",
    "dm_cls": "GoDMEnvPool",
    "gymnasium_cls": "GoGymnasiumEnvPool",
    "max_num_players": 2,
    "komi": 7.5,
    "history_length": 8,
    "max_terminal_steps": 0,
}

register(
    task_id="go_9x9",
    aliases=["PGXGo9x9-v1"],
    board_size=9,
    task="go_9x9",
    **_COMMON,
)

register(
    task_id="go_19x19",
    aliases=["PGXGo19x19-v1"],
    board_size=19,
    task="go_19x19",
    **_COMMON,
)
