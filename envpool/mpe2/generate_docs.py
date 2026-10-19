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
"""Regenerate native/official render comparisons after twelve shared actions."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import envpool.mpe2.registration  # noqa: F401
from envpool.mpe2 import TASKS
from envpool.mpe2.oracle import (
    actions_for,
    make_oracle,
    oracle_actions,
    player_actions,
    render_frame,
    synchronize_reset,
)
from envpool.registration import make_gymnasium


def main() -> None:
    """Write a labeled comparison sheet for every registered scenario."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel_width, panel_height = 720, 405
    sheet = Image.new("RGB", (2 * panel_width, 6 * panel_height), "#e9edf0")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for index, (name, task_id, _) in enumerate(TASKS):
        native = make_gymnasium(task_id, seed=19, render_mode="rgb_array")
        oracle = make_oracle(name)
        try:
            oracle.reset(seed=19)
            _, info = native.reset()
            synchronize_reset(oracle, info)
            rng = np.random.default_rng(23)
            for _ in range(12):
                actions = actions_for(native, info, rng)
                oracle.step(oracle_actions(oracle, actions))
                *_, info = native.step(player_actions(info, actions))
            x, y = index % 2 * panel_width, index // 2 * panel_height
            draw.text((x + 10, y + 5), task_id, font=font, fill="#17212b")
            for column, (label, frame) in enumerate((
                ("EnvPool", render_frame(native)),
                ("MPE2 1.1.1", oracle.render()),
            )):
                draw.text(
                    (x + 10 + column * 350, y + 30),
                    label,
                    font=font,
                    fill="#17212b",
                )
                sheet.paste(
                    Image.fromarray(frame).resize((350, 350)),
                    (x + 10 + column * 350, y + 55),
                )
        finally:
            native.close()
            oracle.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, optimize=True)


if __name__ == "__main__":
    main()
