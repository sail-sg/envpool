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
"""Generate native-versus-official images after real Craftax action sequences."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import envpool.craftax.registration  # noqa: F401
from envpool.craftax.oracle import encode, jax, make_oracle, native, renderer
from envpool.registration import make_gymnasium


def main() -> None:
    """Render representative Classic, overworld, and cave trajectories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panels = []
    for classic, level, length in (
        (True, 0, 96),
        (False, 0, 96),
        (False, 2, 48),
    ):
        task = (
            "Craftax-"
            + ("Classic-" if classic else "")
            + "Symbolic-AutoReset-v1"
        )
        oracle, params = make_oracle(task)
        key = jax.random.PRNGKey(11)
        _, state = oracle.reset(key, params)
        kwargs = {}
        if level:
            state = state.replace(
                player_level=jax.numpy.asarray(level, dtype=jax.numpy.int32),
                player_position=state.up_ladders[level],
            )
            layout = native.Game(native.Params(classic)).get_state()
            kwargs["initial_state"] = encode(state, layout).tolist()
        pool = make_gymnasium(
            task, num_envs=1, seed=11, render_mode="rgb_array", **kwargs
        )
        pool.reset()
        step = jax.jit(oracle.step)
        rng = np.random.default_rng(40)
        try:
            for _ in range(length):
                key, draw = jax.random.split(key)
                action = int(rng.integers(17 if classic else 43))
                _, state, _, _, _ = step(draw, state, action, params)
                pool.step(np.array([action], np.int32))
            actual = np.asarray(pool.render())[0]
            expected = np.asarray(renderer(classic)(state)).astype(np.uint8)
            np.testing.assert_array_equal(actual, expected)
            label = (
                ("Classic" if classic else "Craftax")
                + f" / floor {int(getattr(state, 'player_level', 0))} / {length} actions"
            )
            panels.append((
                label,
                Image.fromarray(actual),
                Image.fromarray(expected),
            ))
        finally:
            pool.close()
    scale = 2
    panel_width = max(image.width for _, image, _ in panels) * scale
    width = panel_width * 2 + 48
    heights = [image.height * scale + 44 for _, image, _ in panels]
    canvas = Image.new("RGB", (width, sum(heights) + 40), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 10), "EnvPool (native C++)", fill=(20, 20, 20))
    draw.text(
        (panel_width + 32, 10), "Official Craftax v1.6.1", fill=(20, 20, 20)
    )
    top = 38
    for (label, left, right), height in zip(panels, heights, strict=True):
        draw.text((16, top), label, fill=(20, 20, 20))
        for column, image in enumerate((left, right)):
            resized = image.resize(
                (image.width * scale, image.height * scale),
                Image.Resampling.NEAREST,
            )
            canvas.paste(resized, (16 + column * (panel_width + 16), top + 22))
        top += height
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)


if __name__ == "__main__":
    main()
