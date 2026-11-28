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
"""Capture three Gymnax scenes with their native and reference renderers."""

import sys
from contextlib import closing

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import envpool.gymnax.registration  # noqa: F401
from envpool.gymnax.oracle import (
    first_slot,
    frame,
    make_oracle,
    stepper,
    synchronize,
)
from envpool.registration import make_gymnasium


def main(output: str) -> None:
    """Render the same state in both columns after several actions."""
    width = 390
    scenes = [
        (
            "SpaceInvaders-MinAtar",
            [3, 2, 2, 0, 0, 3, 1, 1, 1, 1, 3, 2, 0],
            width,
        ),
        ("FourRooms-misc", [0, 1, 1, 1, 1, 1], width),
        ("PointRobot-misc", [[0.1, 0.1]] * 3, width * 3 // 5),
    ]
    height = 56 + sum(h + 40 for _, _, h in scenes)
    sheet = Image.new("RGB", (2 * width + 36, height), "#eef1f4")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=22)
    for x, title in ((12, "EnvPool"), (width + 24, "Gymnax 1.1.0")):
        draw.text(
            (x + width // 2, 24), title, font=font, fill="#1b2837", anchor="mm"
        )
    y = 44
    for name, actions, h in scenes:
        env, params = make_oracle(name)
        step = stepper(env, params)
        with closing(
            make_gymnasium(
                name,
                seed=7,
                render_mode="rgb_array",
                render_width=width,
                render_height=h,
            )
        ) as pool:
            obs, info = pool.reset()
            state = synchronize(env, params, obs[0], first_slot(info))
            for action in np.asarray(
                actions, dtype=pool.spec.action_array_spec["action"].dtype
            ):
                obs, _, _, _, info = pool.step(action[None])
                _, state, *_ = step(state, action, obs[0], first_slot(info))
            rendered = pool.render()
            assert rendered is not None
            draw.text(
                (sheet.width // 2, y + 20),
                name,
                font=font,
                fill="#334459",
                anchor="mm",
            )
            sheet.paste(Image.fromarray(rendered[0]), (12, y + 40))
            sheet.paste(
                Image.fromarray(frame(env, params, state, h, width)),
                (width + 24, y + 40),
            )
        y += h + 40
    sheet.save(output)


if __name__ == "__main__":
    main(sys.argv[1])
