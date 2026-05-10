# ruff: noqa
# fmt: off
# Copyright 2022 InstaDeep Ltd. All rights reserved.
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
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import matplotlib.animation
import matplotlib.cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d
import mpl_toolkits.mplot3d.axes3d
from matplotlib.artist import Artist
from numpy.typing import NDArray

from jumanji.environments.packing.bin_pack.types import State, item_from_space
from jumanji.viewer import MatplotlibViewer


class BinPackViewer(MatplotlibViewer[State]):
    FONT_STYLE = "monospace"

    def __init__(self, name: str, render_mode: str = "human") -> None:
        """Viewer for the `BinPack` environment.

        Args:
            name: the window name to be used when initializing the window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        super().__init__(name, render_mode)

    def render(self, state: State, save_path: Optional[str] = None) -> Optional[NDArray]:
        """Render the given state of the `BinPack` environment.

        Args:
            state: the `State` to render.
            save_path: Optional path to save the rendered environment image to.

        Returns:
            RGB array if the render_mode is 'rgb_array'.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        entities = self._create_entities(state)
        for entity in entities:
            ax.add_collection3d(entity)
        self._add_overlay(fig, ax, state)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        return self._display(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation object that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = self._get_fig_ax(name_suffix="_animation", show=False)
        plt.close(fig=fig)

        entities: List[mpl_toolkits.mplot3d.art3d.Poly3DCollection] = []

        def make_frame(state: State) -> Tuple[Artist]:
            for entity in entities:
                entity.remove()
            entities.clear()
            entities.extend(self._create_entities(state))
            for entity in entities:
                ax.add_collection3d(entity)
            self._add_overlay(fig, ax, state)
            return (ax,)

        # Create the animation object.
        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=states,
            interval=interval,
        )

        # Save the animation as a gif.
        if save_path:
            self._animation.save(save_path)

        return self._animation

    def _get_fig_ax(
        self,
        name_suffix: Optional[str] = None,
        show: bool = True,
        padding: float = 0.05,
        **fig_kwargs: Any,
    ) -> Tuple[plt.Figure, plt.Axes]:
        figure_name = getattr(self, "_figure_name", self._name)
        name = figure_name if name_suffix is None else figure_name + name_suffix
        recreate = not plt.fignum_exists(name)
        fig = plt.figure(name, figsize=self.figure_size)
        if recreate:
            fig.tight_layout()
            if not plt.isinteractive() and show:
                fig.show()
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.get_axes()[0]
        return fig, ax

    def _create_entities(self, state: State) -> List[mpl_toolkits.mplot3d.art3d.Poly3DCollection]:
        entities = []
        n_items = len(state.items_mask)
        cmap = plt.get_cmap("hsv", n_items)
        for i in range(n_items):
            if state.items_placed[i]:
                box = self._create_box(
                    (
                        state.items_location.x[i],
                        state.items_location.y[i],
                        state.items_location.z[i],
                    ),
                    (state.items.x_len[i], state.items.y_len[i], state.items.z_len[i]),
                    cmap(i),
                    0.3,
                )
                entities.append(box)

        container = item_from_space(state.container)
        box = self._create_box(
            (0.0, 0.0, 0.0),
            (container.x_len, container.y_len, container.z_len),
            "cyan",
            0.05,
        )
        entities.append(box)
        return entities

    def _create_box(
        self,
        pos: Tuple[float, float, float],
        lens: Tuple[float, float, float],
        colour: Union[matplotlib.cm.ScalarMappable, str],
        alpha: float,
    ) -> mpl_toolkits.mplot3d.art3d.Poly3DCollection:
        """Add a box to the artist.

        Args:
            pos: (x, y, z)
            lens: lengths for the x, y and z dimensions.
            colour: colour of the box
            alpha: transparency of the box, 0 is completely transparent and 1 is opaque.
        """
        verts = self._create_box_vertices(pos, lens)
        poly3d = mpl_toolkits.mplot3d.art3d.Poly3DCollection(
            verts, linewidths=1, edgecolors="black", facecolors=colour, alpha=alpha
        )
        return poly3d

    def _add_overlay(self, fig: plt.Figure, ax: plt.Axes, state: State) -> None:
        """Sets the bounds of the scene and displays text about the scene.

        Args:
            state: `State` of the environment
        """
        eps = 0.05
        container = item_from_space(state.container)
        ax.set(
            xlim=(-container.x_len * eps, container.x_len * (1 + eps)),
            ylim=(-container.y_len * eps, container.y_len * (1 + eps)),
            zlim=(-container.z_len * eps, container.z_len * (1 + eps)),
        )
        ax.set_xlabel("x", font=self.FONT_STYLE)
        ax.set_ylabel("y", font=self.FONT_STYLE)
        ax.set_zlabel("z", font=self.FONT_STYLE)

        n_items = sum(state.items_mask)
        placed_items = sum(state.items_placed)
        container_volume = float(container.x_len) * float(container.y_len) * float(container.z_len)
        used_volume = self._get_used_volume(state)
        metrics = [
            ("Placed", f"{placed_items:{len(str(n_items))}}/{n_items}"),
            ("Used Volume", f"{used_volume / container_volume:6.1%}"),
        ]
        title = " | ".join(key + ": " + value for key, value in metrics)
        fig.suptitle(title, font=self.FONT_STYLE)

    def _create_box_vertices(
        self, pos: Tuple[float, float, float], lens: Tuple[float, float, float]
    ) -> List[List[Tuple[float, float, float]]]:
        """
        Args:
            pos: (x, y, z) of the corner closest to the origin of the coordinate space.
            lens: the x, y and z lengths of the box.

        Returns:
            For each face composing the box, a list containing the vertices that make that face.
        """
        verts = [
            (pos[0], pos[1], pos[2]),
            (pos[0] + lens[0], pos[1], pos[2]),
            (pos[0] + lens[0], pos[1] + lens[1], pos[2]),
            (pos[0] + lens[0], pos[1] + lens[1], pos[2] + lens[2]),
            (pos[0], pos[1] + lens[1], pos[2] + lens[2]),
            (pos[0], pos[1], pos[2] + lens[2]),
            (pos[0] + lens[0], pos[1], pos[2] + lens[2]),
            (pos[0], pos[1] + lens[1], pos[2]),
        ]
        faces = [
            [0, 1, 2, 7],
            [1, 2, 3, 6],
            [0, 1, 6, 5],
            [0, 7, 4, 5],
            [2, 7, 4, 3],
            [6, 3, 4, 5],
        ]
        return [[verts[i] for i in face] for face in faces]

    def _get_used_volume(self, state: State) -> float:
        used_volume = sum(
            float(state.items.x_len[i]) * float(state.items.y_len[i]) * float(state.items.z_len[i])
            for i, placed in enumerate(state.items_placed)
            if placed
        )
        return used_volume
