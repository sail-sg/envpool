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
"""Assemble the minimal MyoSuite runtime asset tree."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any

_MYODM_OBJECTS = (
    "airplane",
    "alarmclock",
    "apple",
    "banana",
    "binoculars",
    "bowl",
    "camera",
    "coffeemug",
    "cup",
    "cylinderlarge",
    "cylindermedium",
    "cylindersmall",
    "duck",
    "elephant",
    "eyeglasses",
    "flashlight",
    "flute",
    "fryingpan",
    "gamecontroller",
    "gelatinbox",
    "hammer",
    "hand",
    "headphones",
    "knife",
    "lightbulb",
    "mouse",
    "mug",
    "phone",
    "piggybank",
    "pyramidlarge",
    "pyramidmedium",
    "pyramidsmall",
    "rubberduck",
    "scissors",
    "spherelarge",
    "spheremedium",
    "spheresmall",
    "stanfordbunny",
    "stapler",
    "table",
    "teapot",
    "toothbrush",
    "toothpaste",
    "toruslarge",
    "torusmedium",
    "torussmall",
    "train",
    "watch",
    "waterbottle",
    "wineglass",
)
_WINDOWS_SHORT_IMPORT_PACKAGES = ("mujoco",)
_DLL_DIRECTORY_HANDLES: list[Any] = []
mujoco: Any = None


def _copy_short_import_package(
    destination_root: Path, package_name: str
) -> list[Path]:
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.submodule_search_locations is None:
        return []
    source = Path(next(iter(spec.submodule_search_locations)))
    if not source.is_dir():
        return []
    destination = destination_root / package_name
    if not destination.exists():
        shutil.copytree(
            source,
            destination,
            symlinks=False,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    copied = [destination]
    sibling = source.parent / f"{package_name}.libs"
    if sibling.is_dir():
        sibling_destination = destination_root / sibling.name
        if not sibling_destination.exists():
            shutil.copytree(sibling, sibling_destination, symlinks=False)
        copied.append(sibling_destination)
    return copied


def _shorten_windows_binary_imports() -> None:
    if os.name != "nt":
        return
    # Bazel runfiles paths can exceed the Windows DLL loader path limit.
    destination_root = (
        Path(tempfile.gettempdir()) / "envpool_mujoco_runtime_site"
    )
    destination_root.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for package_name in _WINDOWS_SHORT_IMPORT_PACKAGES:
        copied.extend(
            _copy_short_import_package(destination_root, package_name)
        )
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                del sys.modules[name]
    if copied:
        sys.path.insert(0, str(destination_root))
    if hasattr(os, "add_dll_directory"):
        for path in copied:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))


def _load_mujoco() -> Any:
    global mujoco
    if mujoco is None:
        _shorten_windows_binary_imports()
        import mujoco as mujoco_module

        mujoco = mujoco_module
    return mujoco


def _read_manifest(path: Path) -> list[Path]:
    return [Path(line) for line in path.read_text().splitlines() if line]


def _after_marker(path: Path, marker: str) -> str | None:
    text = path.as_posix()
    marker = marker.rstrip("/") + "/"
    index = text.find(marker)
    if index < 0:
        return None
    return text[index + len(marker) :]


def _skip_common(rel: str) -> bool:
    return (
        rel
        in {
            ".gitignore",
            "BUILD.bazel",
            "README.md",
            "REPO.bazel",
            "WORKSPACE",
            "__init__.py",
            "objects.png",
            "preview.py",
            "pyproject.toml",
            "test_sims.py",
        }
        or rel.startswith((".github/", ".idea/", "tests/"))
        or rel.endswith("/object.xml")
        or rel.startswith("scene/")
        and rel.endswith((".mtl", ".obj"))
    )


def _object_needed(rel: str, objects: set[str]) -> bool:
    if rel in {"LICENSE", "common.xml"}:
        return True
    object_dir = rel.split("/", 1)[0]
    return object_dir in objects


def _destination(src: Path, out: Path, objects: set[str]) -> Path | None:
    rel = _after_marker(src, "myosuite_source/myosuite")
    if rel is not None:
        if _skip_common(rel):
            return None
        return out / "myosuite" / rel

    rel = _after_marker(src, "myosuite_mpl_sim")
    if rel is not None:
        keep = rel in {
            "LICENSE",
            "assets/handL_assets.xml",
            "assets/handL_chain.xml",
            "assets/left_arm_assets.xml",
            "assets/left_arm_chain_myochallenge.xml",
        } or rel.startswith("meshes/mplL/")
        if not keep or _skip_common(rel):
            return None
        return out / "myosuite/simhive/MPL_sim" / rel

    rel = _after_marker(src, "myosuite_ycb_sim")
    if rel is not None:
        keep = rel in {
            "LICENSE",
            "includes/assets_009_gelatin_box.xml",
            "includes/body_009_gelatin_box.xml",
            "includes/defaults_ycb.xml",
            "meshes/009_gelatin_box.msh",
            "textures/009_gelatin_box.png",
        }
        if not keep or _skip_common(rel):
            return None
        return out / "myosuite/simhive/YCB_sim" / rel

    rel = _after_marker(src, "myosuite_furniture_sim")
    if rel is not None:
        keep = rel in {
            "LICENSE",
            "common/textures/stone0.png",
            "common/textures/stone1.png",
            "common/textures/wood1.png",
            "simpleTable.xml",
        } or rel.startswith("simpleTable/")
        if not keep or _skip_common(rel):
            return None
        return out / "myosuite/simhive/furniture_sim" / rel

    rel = _after_marker(src, "myosuite_myo_sim")
    if rel is not None:
        if _skip_common(rel):
            return None
        return out / "myosuite/simhive/myo_sim" / rel

    rel = _after_marker(src, "myosuite_object_sim")
    if rel is not None:
        if not _object_needed(rel, objects) or _skip_common(rel):
            return None
        return out / "myosuite/simhive/object_sim" / rel

    return None


def _copy_runtime_sources(out: Path, manifest: Path, objects: set[str]) -> None:
    for src in _read_manifest(manifest):
        dst = _destination(src, out, objects)
        if dst is None or not src.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _generate_myodm_object_xml(out: Path, objects: set[str]) -> None:
    template = out / "myosuite/envs/myo/assets/hand/myohand_object.xml"
    text = template.read_text()
    for object_name in sorted(objects):
        (template.parent / f"myohand_object_{object_name}.xml").write_text(
            text.replace("OBJECT_NAME", object_name)
        )


def _mesh_geoms(body: mujoco.MjsBody) -> list[str]:
    return [
        geom.name
        for geom in body.geoms
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH
    ]


def _apply_arm_reach_edit(spec: mujoco.MjSpec) -> None:
    roots = ("firstmc", "secondmc", "thirdmc", "fourthmc", "fifthmc")
    body_positions: dict[str, list[tuple[str, list[float], list[str]]]] = {}
    for root in roots:
        body_positions[root] = []
        child = spec.body(root).first_body()
        while child is not None:
            body_positions[root].append((
                child.name,
                child.pos.copy(),
                _mesh_geoms(child),
            ))
            child = child.first_body()

    site = spec.site("IFtip")
    site_size = site.size.copy()
    site_pos = site.pos.copy()
    site_rgba = site.rgba.copy()

    for root in roots:
        child = spec.body(root).first_body()
        if child is not None:
            spec.delete(child)

    for root in roots:
        parent = spec.body(root)
        for body_name, pos, mesh_names in body_positions[root]:
            parent.add_body(name=body_name, pos=pos)
            body = spec.body(body_name)
            for mesh_name in mesh_names:
                body.add_geom(
                    meshname=mesh_name,
                    name=body_name,
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                )
            if body_name == "distph2":
                body.add_site(
                    name="IFtip",
                    size=site_size * 2,
                    pos=site_pos,
                    rgba=site_rgba,
                )
            parent = body

    spec.body("world").add_site(
        name="IFtip_target",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.02, 0.02, 0.02],
        pos=[-0.2, -0.2, 1.2],
        rgba=[0.0, 0.0, 1.0, 0.3],
    )


def _generate_arm_reach_xml(out: Path) -> None:
    _load_mujoco()
    myo_sim = (out / "myosuite/simhive/myo_sim").resolve()
    source_arm = myo_sim / "arm"
    with tempfile.TemporaryDirectory(prefix="myosuite-arm-reach-") as temp:
        temp_root = Path(temp)
        spec_root = temp_root / "spec"
        spec_arm = spec_root / "arm"
        shutil.copytree(source_arm, spec_arm, symlinks=True)
        (spec_root / "scene").symlink_to(myo_sim / "scene")
        (spec_root / "myo_sim").symlink_to(myo_sim)
        (spec_arm / "myo_sim").symlink_to(myo_sim)
        (temp_root / "myo_sim").symlink_to(myo_sim)

        spec = mujoco.MjSpec.from_file(str(spec_arm / "myoarm.xml"))
        _apply_arm_reach_edit(spec)
        spec.compile()
        (source_arm / "myoarm_reach.xml").write_text(spec.to_xml())


def _recursive_immobilize(
    spec: mujoco.MjSpec,
    temp_model: mujoco.MjModel,
    parent: mujoco.MjsBody,
    remove_eqs: bool = False,
    remove_actuators: bool = False,
) -> list[int]:
    removed_joint_ids: list[int] = []
    for site in list(parent.sites):
        spec.delete(site)
    for joint in list(parent.joints):
        removed_joint_ids.extend(temp_model.joint(joint.name).qposadr)
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
                spec, temp_model, child, remove_eqs, remove_actuators
            )
        )
    return removed_joint_ids


def _recursive_remove_contacts(
    parent: mujoco.MjsBody,
    return_condition: Callable[[mujoco.MjsBody], bool] | None = None,
) -> None:
    if return_condition is not None and return_condition(parent):
        return
    for geom in parent.geoms:
        geom.contype = 0
        geom.conaffinity = 0
    for child in parent.bodies:
        _recursive_remove_contacts(child, return_condition)


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
            spec_copy.delete(child)
            continue
        _recursive_mirror(meshes_to_mirror, spec_copy, child)


def _preprocess_tabletennis_spec(spec: mujoco.MjSpec) -> mujoco.MjSpec:
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
    removed = set(removed_ids)
    for key in spec.keys:
        key.qpos = [
            value for idx, value in enumerate(key.qpos) if idx not in removed
        ]

    _recursive_remove_contacts(
        spec.body("full_body"),
        return_condition=lambda body: "radius" in body.name,
    )

    torso = spec.body("torso")
    spec_copy = spec.copy()
    attachment_frame = torso.add_frame(
        quat=[0.5, 0.5, -0.5, 0.5],
        pos=[0.05, 0.373, -0.04],
    )
    for collection in (
        spec_copy.keys,
        spec_copy.textures,
        spec_copy.materials,
        spec_copy.tendons,
        spec_copy.actuators,
        spec_copy.equalities,
        spec_copy.sensors,
        spec_copy.cameras,
    ):
        for item in list(collection):
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
    return spec


def _generate_tabletennis_xml(out: Path) -> None:
    _load_mujoco()
    asset_arm = out / "myosuite/envs/myo/assets/arm"
    source = asset_arm / "myoarm_tabletennis.xml"
    spec = mujoco.MjSpec.from_file(str(source))
    _preprocess_tabletennis_spec(spec)
    xml_text = _normalize_tabletennis_xml(spec.to_xml())
    (asset_arm / "myoarm_tabletennis_native.xml").write_text(xml_text)


def _normalize_tabletennis_xml(xml_text: str) -> str:
    root = ET.fromstring(xml_text)
    seen_default_classes: set[str] = set()

    def visit(parent: ET.Element) -> None:
        index = 0
        while index < len(parent):
            child = parent[index]
            if (
                parent.tag == "default"
                and child.tag == "default"
                and not child.attrib
            ):
                grandchildren = list(child)
                parent.remove(child)
                for offset, grandchild in enumerate(grandchildren):
                    parent.insert(index + offset, grandchild)
                continue
            elif child.tag == "default" and "class" in child.attrib:
                class_name = child.attrib["class"]
                if class_name in seen_default_classes:
                    parent.remove(child)
                else:
                    seen_default_classes.add(class_name)
                    visit(child)
                    index += 1
            else:
                visit(child)
                index += 1

    visit(root)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("objects", type=Path)
    return parser.parse_args()


def main() -> None:
    """Generate the minimal runtime asset tree used by native MyoSuite."""
    args = _parse_args()
    objects = set(args.objects.read_text().splitlines()) or set(_MYODM_OBJECTS)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "myosuite/simhive").mkdir(parents=True, exist_ok=True)
    _copy_runtime_sources(args.out, args.manifest, objects)
    _generate_myodm_object_xml(args.out, objects)
    _generate_arm_reach_xml(args.out)
    _generate_tabletennis_xml(args.out)


if __name__ == "__main__":
    main()
