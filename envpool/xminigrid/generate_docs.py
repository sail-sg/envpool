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
"""Render the three stages of the XLand recipe against the pinned oracle."""

import sys
from contextlib import closing

import jax
import numpy as np
import xminigrid
from PIL import Image, ImageDraw, ImageFont

import envpool.xminigrid.registration  # noqa: F401
from envpool.registration import make_gymnasium
from envpool.xminigrid.oracle import (
    RECIPE,
    oracle_ruleset,
    recipe_actions,
    synchronize_reset,
)


def main(output: str) -> None:
    """Save the native and upstream views of a solved episode side by side."""
    name = "XLand-MiniGrid-R1-9x9"
    env, p = xminigrid.make(name, ruleset=oracle_ruleset(RECIPE))
    step = jax.jit(env.step)
    size = p.width * 32
    sheet = Image.new("RGB", (2 * size + 36, 3 * (size + 40) + 52), "#eef1f4")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for col, label in enumerate(("EnvPool", "xminigrid 0.9.3")):
        draw.text(
            (12 + col * (size + 12) + size // 2, 24),
            label,
            fill="#1b2837",
            font=font,
            anchor="mm",
        )
    with closing(
        make_gymnasium(name, ruleset=RECIPE, seed=63, render_mode="rgb_array")
    ) as pool:
        _, info = pool.reset()
        t = synchronize_reset(name, env.reset(p, jax.random.key(0)), info)

        def capture(row: int, title: str) -> None:
            y = 40 + row * (size + 40)
            draw.text(
                (sheet.width // 2, y + 20),
                title,
                font=font,
                anchor="mm",
                fill="#334459",
            )
            frame = pool.render()
            assert frame is not None
            for col, img in enumerate((
                frame[0],
                env.render(p, jax.device_get(t)),
            )):
                sheet.paste(
                    Image.fromarray(img), (12 + col * (size + 12), y + 40)
                )

        capture(0, "Initial objects")
        for action in recipe_actions(info):
            _, reward, term, _, _ = pool.step(np.array([action], np.int32))
            t = step(p, t, np.int32(action))
            if action == 4:
                if term[0]:
                    capture(2, "Red ball beside green ball")
                else:
                    capture(1, "Pyramid and square become a red ball")
        assert reward[0] > 0 and t.reward > 0
    sheet.save(output)


if __name__ == "__main__":
    main(sys.argv[1])
