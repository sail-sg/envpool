#!/usr/bin/env python3
"""Smoke-check that release tests import installed wheels, including assets."""

from __future__ import annotations

import argparse
import importlib
from importlib.metadata import requires, version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

_ASSET_PACKAGES = {
    "envpool-assets-mjlab": ("envpool_assets_mjlab", []),
    "envpool-assets": (
        "envpool_assets",
        [
            "atari/roms/pong.bin",
            "gfootball/assets/data",
            "mujoco/assets_dmc/cartpole.xml",
            "mujoco/assets_gym/ant.xml",
            "mujoco/locomotion/assets_dm_control/mocap_2019.bin",
            "mujoco/locomotion/assets_dm_control/mocap_2020.bin",
            "mujoco/locomotion/assets_labmaze/style_01/wall_blue_d.png",
            "mujoco/metaworld/assets/objects/assets/drawer.xml",
            "mujoco/robotics/assets/fetch/reach.xml",
            "procgen/assets/platformer/playerBlue_dead.png",
            "vizdoom/bin/freedoom2.wad",
            "vizdoom/maps/basic.wad",
        ],
    ),
    "envpool-assets-mujoco-large": (
        "envpool_assets_mujoco_large",
        [
            "mujoco/myosuite/assets/myosuite/envs/myo/assets/hand/myohand_pose.xml",
            "mujoco/playground/assets/mujoco_menagerie/google_barkour_vb/scene_mjx.xml",
            (
                "mujoco/playground/assets/mujoco_playground/_src/"
                "locomotion/go1/xmls/scene_mjx_feetonly_flat_terrain.xml"
            ),
            (
                "mujoco/playground/assets/mujoco_menagerie/"
                "unitree_go1/assets/trunk.stl"
            ),
            (
                "mujoco/playground/assets/mujoco_playground/_src/"
                "locomotion/spot/xmls/scene_mjx_feetonly_flat_terrain.xml"
            ),
            "mujoco/playground/assets/mujoco_menagerie/boston_dynamics_spot/spot.xml",
            (
                "mujoco/playground/assets/mujoco_playground/_src/"
                "manipulation/aero_hand/xmls/scene_mjx_cube.xml"
            ),
            (
                "mujoco/playground/assets/mujoco_menagerie/"
                "tetheria_aero_hand_open/right_hand.xml"
            ),
            (
                "mujoco/playground/assets/mujoco_playground/_src/"
                "manipulation/aloha/xmls/mjx_hand_over.xml"
            ),
            "mujoco/playground/assets/mujoco_menagerie/aloha/aloha.xml",
            (
                "mujoco/playground/assets/mujoco_playground/_src/"
                "manipulation/franka_emika_panda/xmls/mjx_single_cube.xml"
            ),
            "mujoco/playground/assets/mujoco_menagerie/franka_emika_panda/panda.xml",
            (
                "mujoco/playground/assets/mujoco_playground/_src/"
                "manipulation/franka_emika_panda_robotiq/xmls/"
                "scene_panda_robotiq_cube.xml"
            ),
            "mujoco/playground/assets/mujoco_menagerie/franka_emika_panda/panda.xml",
            "mujoco/playground/assets/mujoco_menagerie/robotiq_2f85_v4/mjx_2f85.xml",
            (
                "mujoco/playground/assets/mujoco_playground/_src/"
                "manipulation/leap_hand/xmls/scene_mjx_cube.xml"
            ),
            "mujoco/playground/assets/mujoco_menagerie/leap_hand/right_hand.xml",
        ],
    ),
    "envpool-assets-mujoco-playground-humanoid": (
        "envpool_assets_mujoco_playground_humanoid",
        [
            (
                "mujoco/playground/assets/mujoco_playground/_src/"
                "locomotion/apollo/xmls/scene_mjx_feetonly_flat_terrain.xml"
            ),
            (
                "mujoco/playground/assets/mujoco_menagerie/"
                "apptronik_apollo/apptronik_apollo.xml"
            ),
            (
                "mujoco/playground/assets/mujoco_playground/_src/"
                "locomotion/berkeley_humanoid/xmls/"
                "scene_mjx_feetonly_flat_terrain.xml"
            ),
            (
                "mujoco/playground/assets/mujoco_menagerie/"
                "berkeley_humanoid/berkeley_humanoid.xml"
            ),
            (
                "mujoco/playground/assets/mujoco_playground/_src/"
                "locomotion/g1/xmls/scene_mjx_feetonly_flat_terrain.xml"
            ),
            "mujoco/playground/assets/mujoco_menagerie/unitree_g1/g1_mjx.xml",
            (
                "mujoco/playground/assets/mujoco_playground/_src/"
                "locomotion/h1/xmls/scene_mjx_feetonly.xml"
            ),
            "mujoco/playground/assets/mujoco_menagerie/unitree_h1/h1.xml",
            (
                "mujoco/playground/assets/mujoco_playground/_src/"
                "locomotion/op3/xmls/scene_mjx_feetonly.xml"
            ),
            "mujoco/playground/assets/mujoco_menagerie/robotis_op3/op3.xml",
            (
                "mujoco/playground/assets/mujoco_playground/_src/"
                "locomotion/t1/xmls/scene_mjx_feetonly_flat_terrain.xml"
            ),
            "mujoco/playground/assets/mujoco_menagerie/booster_t1/t1.xml",
        ],
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="EnvPool source checkout that must not be imported.",
    )
    return parser.parse_args()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _check_asset_package(
    distribution: str, module_name: str, rels: list[str]
) -> None:
    asset_version = Version(version(distribution))
    requirement = next(
        item
        for item in map(Requirement, requires("envpool") or [])
        if item.name == distribution
    )
    if asset_version not in requirement.specifier:
        raise RuntimeError(
            f"unexpected {distribution} version: {asset_version}"
        )
    module = importlib.import_module(module_name)
    asset_root = Path(module.asset_path()).resolve()
    if distribution == "envpool-assets-mjlab":
        from envpool.mujoco.mjlab import TASKS

        rels = [
            f"mujoco/mjlab/assets/{task['asset']}/{filename}"
            for task in TASKS.values()
            for filename in ("task.json", "model.mjb.json", "physics.wrp.json")
        ]
    missing = [
        asset_root / rel for rel in rels if not (asset_root / rel).exists()
    ]
    if missing:
        raise RuntimeError(f"{distribution} missing required files: {missing}")
    print(f"{distribution} {asset_version} installed at {asset_root}")


def main() -> None:
    """Check installed EnvPool and envpool-assets package wiring."""
    args = _parse_args()
    source_root = args.source_root.resolve()

    import envpool
    from envpool.registration import package_base_path

    envpool_package = Path(envpool.__file__).resolve().parent

    if _is_relative_to(envpool_package, source_root):
        raise RuntimeError(
            f"imported EnvPool from source tree: {envpool_package}"
        )

    if Path(package_base_path).resolve() != envpool_package:
        raise RuntimeError(
            "EnvPool package_base_path is "
            f"{package_base_path}, expected {envpool_package}"
        )

    bundled_locomotion_assets = list(
        (envpool_package / "mujoco/locomotion").glob("assets_*")
    )
    if bundled_locomotion_assets:
        raise RuntimeError(
            "locomotion data must ship in envpool-assets, not envpool: "
            f"{bundled_locomotion_assets}"
        )

    if (envpool_package / "mujoco/mjlab/assets").exists():
        raise RuntimeError("MJLab data must ship in envpool-assets-mjlab")

    for distribution, (module_name, rels) in _ASSET_PACKAGES.items():
        _check_asset_package(distribution, module_name, rels)

    required_package_files = [
        envpool_package / "mujoco/locomotion/licenses/LICENSE",
        envpool_package / "mujoco/locomotion/licenses/LICENSE.scipy",
        envpool_package / "mujoco/locomotion/licenses/NOTICE",
        envpool_package / "vizdoom/bin/vizdoom",
        envpool_package / "vizdoom/bin/vizdoom.pk3",
    ]
    missing = [path for path in required_package_files if not path.exists()]
    if missing:
        raise RuntimeError(f"envpool wheel missing required files: {missing}")

    print(f"envpool installed at {envpool_package}")


if __name__ == "__main__":
    main()
