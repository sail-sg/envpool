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

"""Generate the vendored MyoSuite table-tennis runtime XML asset."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco


def _recursive_immobilize(
    spec: mujoco.MjSpec,
    temp_model: mujoco.MjModel,
    parent: mujoco.MjsBody,
    *,
    remove_eqs: bool = False,
    remove_actuators: bool = False,
) -> list[int]:
    removed_joint_ids: list[int] = []
    for site in list(parent.sites):
        spec.delete(site)
    for joint in list(parent.joints):
        removed_joint_ids.extend(list(temp_model.joint(joint.name).qposadr))
        if remove_eqs:
            for equality in list(spec.equalities):
                if equality.type == mujoco.mjtEq.mjEQ_JOINT and (
                    equality.name1 == joint.name or equality.name2 == joint.name
                ):
                    spec.delete(equality)
        if remove_actuators:
            for actuator in list(spec.actuators):
                if (
                    actuator.trntype == mujoco.mjtTrn.mjTRN_JOINT
                    and actuator.target == joint.name
                ):
                    spec.delete(actuator)
        spec.delete(joint)
    for child in list(parent.bodies):
        removed_joint_ids.extend(
            _recursive_immobilize(
                spec,
                temp_model,
                child,
                remove_eqs=remove_eqs,
                remove_actuators=remove_actuators,
            )
        )
    return removed_joint_ids


def _recursive_remove_contacts(
    parent: mujoco.MjsBody,
    *,
    return_condition: Callable[[mujoco.MjsBody], bool] | None = None,
) -> None:
    if return_condition is not None and return_condition(parent):
        return
    for geom in list(parent.geoms):
        geom.contype = 0
        geom.conaffinity = 0
    for child in list(parent.bodies):
        _recursive_remove_contacts(child, return_condition=return_condition)


def _recursive_mirror(
    meshes_to_mirror: set[str],
    spec_copy: mujoco.MjSpec,
    parent: mujoco.MjsBody,
) -> None:
    parent.pos[1] *= -1
    parent.quat[[1, 3]] *= -1
    parent.name += "_mirrored"
    for geom in list(parent.geoms):
        if geom.type != mujoco.mjtGeom.mjGEOM_MESH:
            spec_copy.delete(geom)
            continue
        geom.pos[1] *= -1
        geom.quat[[1, 3]] *= -1
        geom.name += "_mirrored"
        geom.group = 1
        meshes_to_mirror.add(geom.meshname)
        geom.meshname += "_mirrored"
    for child in list(parent.bodies):
        if "ping_pong" in child.name:
            spec_copy.detach_body(child)
            continue
        _recursive_mirror(meshes_to_mirror, spec_copy, child)


def _tabletennis_spec_to_xml(model_path: str) -> str:
    spec = mujoco.MjSpec.from_file(model_path)
    for sensor in list(spec.sensors):
        if "pingpong" not in sensor.name and "paddle" not in sensor.name:
            spec.delete(sensor)

    temp_model = spec.compile()
    removed_ids = _recursive_immobilize(
        spec, temp_model, spec.body("femur_l"), remove_eqs=True
    )
    removed_ids.extend(
        _recursive_immobilize(
            spec, temp_model, spec.body("femur_r"), remove_eqs=True
        )
    )
    for key in list(spec.keys):
        key.qpos = [
            qpos for idx, qpos in enumerate(key.qpos) if idx not in removed_ids
        ]

    _recursive_remove_contacts(
        spec.body("full_body"),
        return_condition=lambda body: "radius" in body.name,
    )

    spec_copy = spec.copy()
    torso = spec.body("torso")
    attachment_frame = torso.add_frame(
        quat=[0.5, 0.5, -0.5, 0.5],
        pos=[0.05, 0.373, -0.04],
    )
    for collection in (
        list(spec_copy.keys),
        list(spec_copy.textures),
        list(spec_copy.materials),
        list(spec_copy.tendons),
        list(spec_copy.actuators),
        list(spec_copy.equalities),
        list(spec_copy.sensors),
        list(spec_copy.cameras),
    ):
        for item in collection:
            spec_copy.delete(item)
    _recursive_immobilize(spec_copy, temp_model, spec_copy.worldbody)
    _recursive_remove_contacts(spec_copy.worldbody)
    meshes_to_mirror: set[str] = set()
    _recursive_mirror(meshes_to_mirror, spec_copy, spec_copy.body("clavicle"))
    for mesh in list(spec_copy.meshes):
        if mesh.name in meshes_to_mirror:
            mesh.name += "_mirrored"
            mesh.scale[1] *= -1
        else:
            spec_copy.delete(mesh)
    attachment_frame.attach_body(spec_copy.body("clavicle_mirrored"))
    spec.body("ulna_mirrored").quat = [0.546, 0, 0, -0.838]
    spec.body("humerus_mirrored").quat = [0.924, 0.383, 0, 0]

    root = ET.fromstring(spec.to_xml())
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", ".")
        compiler.set("texturedir", ".")
    defaults = root.find("default")
    if defaults is not None:
        defaults.set("class", "main")
        for child in list(defaults):
            if child.tag == "default" and "class" not in child.attrib:
                defaults.remove(child)
    for elem in root.iter():
        file_attr = elem.get("file")
        if file_attr is None:
            continue
        if file_attr.startswith("../myo_sim/"):
            elem.set("file", "../" + file_attr[len("../myo_sim/") :])
        elif file_attr.startswith("../../envs/"):
            elem.set("file", "../" + file_attr)
    return ET.tostring(root, encoding="unicode")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Generate the normalized runtime XML asset from the upstream model."""
    args = _parse_args()
    xml = _tabletennis_spec_to_xml(str(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(xml, encoding="utf-8")


if __name__ == "__main__":
    main()
