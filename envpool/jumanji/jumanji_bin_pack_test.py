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
"""Native BinPack rule tests."""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

import envpool.jumanji.registration  # noqa: F401
from envpool.registration import make_gymnasium


class JumanjiBinPackTest(absltest.TestCase):
    """Checks native BinPack transitions."""

    def test_place_item_in_empty_space(self) -> None:
        """Checks placement updates item and EMS state."""
        lengths = ",".join(map(str, [0.5] * 8 + [0] * 12))
        env = make_gymnasium(
            "BinPack-v2",
            num_envs=1,
            seed=0,
            render_mode="rgb_array",
            bin_pack_item_x_len=lengths,
            bin_pack_item_y_len=lengths,
            bin_pack_item_z_len=lengths,
            bin_pack_items_mask=",".join(map(str, [1] * 8 + [0] * 12)),
        )
        try:
            obs, _ = env.reset()
            self.assertTrue(bool(obs["ems_mask"][0, 0]))
            self.assertAlmostEqual(float(obs["ems"]["x1"][0, 0]), 0.0)
            self.assertAlmostEqual(float(obs["items"]["x_len"][0, 0]), 0.5)
            self.assertTrue(bool(obs["items_mask"][0, 0]))
            self.assertTrue(bool(obs["action_mask"][0, 0, 0]))

            obs, reward, terminated, truncated, _ = env.step(
                np.asarray([[0, 0]], dtype=np.int32)
            )
            self.assertAlmostEqual(float(reward[0]), 0.125, places=6)
            self.assertFalse(bool(terminated[0]))
            self.assertFalse(bool(truncated[0]))
            self.assertTrue(bool(obs["items_placed"][0, 0]))
            self.assertTrue(bool(obs["items_mask"][0, 0]))
            self.assertFalse(bool(obs["action_mask"][0, 0, 0]))
            self.assertTrue(bool(np.any(obs["action_mask"][0])))

            total_reward = float(reward[0])
            for item in range(1, 8):
                valid_spaces = np.flatnonzero(obs["action_mask"][0, :, item])
                self.assertNotEmpty(valid_spaces)
                obs, reward, terminated, truncated, _ = env.step(
                    np.asarray([[valid_spaces[0], item]], dtype=np.int32)
                )
                total_reward += float(reward[0])
                self.assertEqual(bool(terminated[0]), item == 7)
                self.assertFalse(bool(truncated[0]))
            self.assertAlmostEqual(total_reward, 1.0)
            self.assertFalse(bool(obs["action_mask"].any()))

            frame = env.render(env_ids=np.asarray([0], dtype=np.int32))

            assert frame is not None
            self.assertEqual(frame.shape, (1, 256, 256, 3))
            self.assertGreater(int(frame.max() - frame.min()), 0)
        finally:
            env.close()

    def test_render_preserves_item_dimensions(self) -> None:
        """Different box dimensions must not render as equal count markers."""
        widths = []
        for length in (0.2, 0.7):
            env = make_gymnasium(
                "BinPack-v2",
                num_envs=1,
                seed=0,
                render_mode="rgb_array",
                bin_pack_item_x_len=",".join(map(str, [length] * 20)),
                bin_pack_item_y_len=",".join(["0.2"] * 20),
                bin_pack_item_z_len=",".join(["0.2"] * 20),
                bin_pack_items_mask=",".join(["1"] + ["0"] * 19),
            )
            try:
                env.reset()
                env.step(np.asarray([[0, 0]], dtype=np.int32))
                rendered = env.render()
                assert rendered is not None
                frame = rendered[0].astype(np.int16)
                # The container is gray; isolate the colored placed box.
                rows, columns = np.nonzero(np.ptp(frame, axis=-1) > 30)
                self.assertNotEmpty(rows)
                widths.append(np.ptp(columns))
            finally:
                env.close()
        self.assertGreater(widths[1], 2 * widths[0])


if __name__ == "__main__":
    absltest.main()
