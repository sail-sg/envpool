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
"""Pinned oracle helpers, imported only by tests and documentation tooling."""

import ast
import inspect
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(
        Path(os.environ.get("TEST_TMPDIR", tempfile.gettempdir()))
        / "craftax-mpl"
    ),
)

import envpool.craftax.craftax_debug as native
import jax
import numpy as np
from craftax.craftax import constants as full_constants
from craftax.craftax import renderer as full_renderer
from craftax.craftax_classic import constants as classic_constants
from craftax.craftax_classic import renderer as classic_renderer
from craftax.craftax_env import make_craftax_env_from_name

jax.config.update("jax_threefry_partitionable", True)
jax.config.update("jax_enable_x64", False)

# Only the cache location changes. All textures and calculations come from the
# pinned source; the Bazel runfiles themselves remain read-only.
_cache = (
    Path(os.environ.get("TEST_TMPDIR", tempfile.gettempdir()))
    / "craftax-v1.6.1-textures"
)
_cache.mkdir(parents=True, exist_ok=True)
_built_cache = Path(__file__).parent / "oracle_textures"
if _built_cache.is_dir():
    _cache = _built_cache
full_constants.TEXTURE_CACHE_FILE = str(_cache / "full.bz2")
classic_constants.TEXTURE_CACHE_FILE = str(_cache / "classic.bz2")


def flatten(state: Any) -> dict[str, np.ndarray]:
    """Flatten the official dataclasses without changing values or shapes."""
    result: dict[str, np.ndarray] = {}

    def visit(value: Any, prefix: str = "") -> None:
        if hasattr(value, "__dataclass_fields__"):
            for field in value.__dataclass_fields__:
                if field != "fractal_noise_angles":
                    visit(getattr(value, field), prefix + field + ".")
        else:
            result[prefix[:-1]] = np.asarray(value)

    visit(state)
    return result


def encode(state: Any, layout: Any) -> np.ndarray:
    """Serialize official values in the native diagnostic traversal order."""
    flat = flatten(state)
    assert flat.keys() == layout.keys()
    return np.concatenate([
        flat[key].reshape(-1).astype(np.float64) for key in layout
    ])


def make_oracle(task_id: str, max_steps: int = 257) -> tuple[Any, Any]:
    """Construct the pinned factory name with the requested episode limit."""
    env = make_craftax_env_from_name(
        task_id, auto_reset="-AutoReset-" in task_id
    )
    return env, env.default_params.replace(max_timesteps=max_steps)


def factory_names() -> tuple[str, ...]:
    """Discover names in the oracle factory so coverage cannot omit new tasks."""
    tree = ast.parse(inspect.getsource(make_craftax_env_from_name))
    return tuple(
        sorted({
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("Craftax-")
            and node.value.endswith("-v1")
        })
    )


@lru_cache(maxsize=4)
def renderer(classic: bool, tile: int = 16) -> Any:
    """Compile the unchanged official renderer at one supported tile size."""
    module = classic_renderer if classic else full_renderer
    return jax.jit(module.make_craftax_pixel_renderer(tile))


def reset_info(state: Any, classic: bool) -> dict[str, Any]:
    """Construct the official information values for a nonterminal reset."""
    from craftax.craftax.envs.common import log_achievements_to_info
    from craftax.craftax_classic.envs.common import compute_score

    info = (compute_score if classic else log_achievements_to_info)(
        state, False
    )
    return dict(info, discount=1.0)


__all__ = [
    "native",
    "jax",
    "full_constants",
    "classic_constants",
    "flatten",
    "encode",
    "make_oracle",
    "factory_names",
    "renderer",
    "reset_info",
]
