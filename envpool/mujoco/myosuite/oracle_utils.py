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
"""Shared MyoSuite oracle helpers for tests and docs generation."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any, Iterator

from envpool.mujoco.myosuite.paths import (
    myosuite_asset_root,
    resolve_workspace_path,
)


def _replace_all(text: str, old: str, new: str) -> str:
    return text.replace(old, new)


def _relative_model_path(path: Path, *, oracle_dir: Path) -> str:
    prefix = "\\" if os.name == "nt" else "/"
    return prefix + os.path.relpath(path, oracle_dir)


def find_vendored_myosuite_root() -> Path:
    """Locate the vendored upstream MyoSuite Python source tree."""
    root = resolve_workspace_path(".")
    for candidate in (root, *root.parents):
        direct = candidate / "myosuite_src"
        if (direct / "myosuite/envs/myo/myobase/pose_v0.py").exists():
            return direct
        for pose_path in candidate.rglob(
            "myosuite/envs/myo/myobase/pose_v0.py"
        ):
            return pose_path.parents[4]
    raise FileNotFoundError("Unable to locate vendored myosuite source root")


@cache
def prepare_oracle_imports() -> None:
    """Expose vendored MyoSuite Python modules on sys.path."""
    source_root = str(find_vendored_myosuite_root())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)


@cache
def load_oracle_class(entry_module: str, class_name: str) -> Any:
    """Import one official MyoSuite environment class from vendored source."""
    prepare_oracle_imports()
    module = importlib.import_module(entry_module)
    return getattr(module, class_name)


@contextmanager
def prepared_track_oracle_model_path() -> Iterator[str]:
    """Yield a TrackEnv model_path that resolves to writable staged assets."""
    asset_root = myosuite_asset_root()
    source_model = asset_root / "envs/myo/assets/hand/myohand_object.xml"
    object_xml = source_model.read_text()
    tabletop_xml = (
        asset_root / "envs/myo/assets/hand/myohand_tabletop.xml"
    ).read_text()
    hand_assets_xml = (
        asset_root / "simhive/myo_sim/hand/assets/myohand_assets.xml"
    ).read_text()
    myo_sim_root = asset_root / "simhive/myo_sim"
    myo_sim_root_str = str(myo_sim_root)
    object_sim_root = asset_root / "simhive/object_sim"
    object_sim_root_str = str(object_sim_root)
    oracle_dir = find_vendored_myosuite_root() / "myosuite/envs/myo/myodm"

    with tempfile.TemporaryDirectory(prefix="envpool_myodm_oracle_") as td:
        tmp_dir = Path(td)
        hand_assets_tmp = tmp_dir / "myohand_assets.xml"
        tabletop_tmp = tmp_dir / "myohand_tabletop.xml"
        object_tmp = tmp_dir / "myohand_object.xml"

        hand_assets_xml = _replace_all(
            hand_assets_xml,
            'meshdir=".." texturedir=".."',
            f'meshdir="{myo_sim_root_str}" texturedir="{myo_sim_root_str}"',
        )
        hand_assets_tmp.write_text(hand_assets_xml)

        tabletop_xml = _replace_all(
            tabletop_xml,
            "../../../../simhive/myo_sim/hand/assets/myohand_assets.xml",
            str(hand_assets_tmp),
        )
        tabletop_xml = _replace_all(
            tabletop_xml,
            "../../../../simhive/furniture_sim/simpleTable/simpleTable_asset.xml",
            str(
                asset_root
                / "simhive/furniture_sim/simpleTable/simpleTable_asset.xml"
            ),
        )
        tabletop_xml = _replace_all(
            tabletop_xml,
            "../../../../simhive/myo_sim/hand/assets/myohand_body.xml",
            str(asset_root / "simhive/myo_sim/hand/assets/myohand_body.xml"),
        )
        tabletop_xml = _replace_all(
            tabletop_xml,
            "../../../../simhive/furniture_sim/simpleTable/"
            "simpleGraniteTable_body.xml",
            str(
                asset_root
                / "simhive/furniture_sim/simpleTable/simpleGraniteTable_body.xml"
            ),
        )
        tabletop_xml = _replace_all(
            tabletop_xml,
            'meshdir="../../../../simhive/myo_sim/" texturedir="../../../../simhive/myo_sim/"',
            f'meshdir="{myo_sim_root_str}" texturedir="{myo_sim_root_str}"',
        )
        tabletop_tmp.write_text(tabletop_xml)

        object_xml = _replace_all(
            object_xml, "myohand_tabletop.xml", str(tabletop_tmp)
        )
        object_xml = _replace_all(
            object_xml,
            "../../../../simhive/object_sim/common.xml",
            str(object_sim_root / "common.xml"),
        )
        object_xml = _replace_all(
            object_xml,
            "../../../../simhive/object_sim/OBJECT_NAME/assets.xml",
            object_sim_root_str + "/OBJECT_NAME/assets.xml",
        )
        object_xml = _replace_all(
            object_xml,
            "../../../../simhive/object_sim/OBJECT_NAME/body.xml",
            object_sim_root_str + "/OBJECT_NAME/body.xml",
        )
        object_tmp.write_text(object_xml)

        yield _relative_model_path(object_tmp, oracle_dir=oracle_dir)
