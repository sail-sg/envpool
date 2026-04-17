#!/usr/bin/env python3
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

"""Generate the vendored MyoSuite arm-reaching runtime XML asset."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco


def _first_child_body(body: mujoco.MjsBody) -> mujoco.MjsBody | None:
    return body.first_body()


def _arm_reaching_spec_to_xml(model_path: str) -> str:
    spec = mujoco.MjSpec.from_file(model_path)
    root_names = ("firstmc", "secondmc", "thirdmc", "fourthmc", "fifthmc")
    tip_site = spec.site("IFtip")
    tip_site_name = str(tip_site.name)
    tip_site_size = tip_site.size.copy()
    tip_site_pos = tip_site.pos.copy()
    tip_site_rgba = tip_site.rgba.copy()
    body_chains: dict[str, list[tuple[str, object, list[str]]]] = {}

    for root_name in root_names:
        body_chains[root_name] = []
        root_body = spec.body(root_name)
        child = _first_child_body(root_body)
        while child is not None:
            mesh_names = [
                str(geom.name)
                for geom in child.geoms
                if geom.type == mujoco.mjtGeom.mjGEOM_MESH
            ]
            body_chains[root_name].append((
                str(child.name),
                child.pos.copy(),
                mesh_names,
            ))
            child = _first_child_body(child)

    for root_name in root_names:
        root_body = spec.body(root_name)
        child = _first_child_body(root_body)
        if child is not None:
            spec.delete(child)

    for root_name in root_names:
        parent = spec.body(root_name)
        for body_name, pos, mesh_names in body_chains[root_name]:
            parent.add_body(name=body_name, pos=pos)
            current = spec.body(body_name)
            for mesh_name in mesh_names:
                current.add_geom(
                    meshname=mesh_name,
                    name=body_name,
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                )
            if body_name == "distph2":
                current.add_site(
                    name=tip_site_name,
                    size=tip_site_size * 2.0,
                    pos=tip_site_pos,
                    rgba=tip_site_rgba,
                )
            parent = current

    spec.body("world").add_site(
        name="IFtip_target",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.02, 0.02, 0.02],
        pos=[-0.2, -0.2, 1.2],
        rgba=[0.0, 0.0, 1.0, 0.3],
    )
    return spec.to_xml()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Generate the checked-in arm-reaching XML next to the vendored assets."""
    args = _parse_args()
    with tempfile.TemporaryDirectory() as td:
        staged_root = Path(td) / "simhive"
        staged_root.mkdir(parents=True, exist_ok=True)
        (staged_root / "myo_sim").symlink_to(
            args.input.resolve().parent.parent,
            target_is_directory=True,
        )
        staged_model = staged_root / "myo_sim" / "arm" / args.input.name
        xml_tree = ET.ElementTree(
            ET.fromstring(_arm_reaching_spec_to_xml(str(staged_model)))
        )
    root = xml_tree.getroot()
    if root is not None:
        compiler = root.find("compiler")
        if compiler is not None:
            compiler.set("meshdir", ".")
            compiler.set("texturedir", ".")
        for elem in root.iter():
            file_attr = elem.get("file")
            if file_attr is not None and file_attr.startswith("../myo_sim/"):
                elem.set("file", "../" + file_attr[len("../myo_sim/"):])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    xml_tree.write(args.output, encoding="utf-8")


if __name__ == "__main__":
    main()
