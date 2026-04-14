#!/usr/bin/env python3
"""Smoke-check that release tests import installed wheels, including assets."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

from packaging.version import Version


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


def main() -> None:
    """Check installed EnvPool and envpool-assets package wiring."""
    args = _parse_args()
    source_root = args.source_root.resolve()

    import envpool_assets

    import envpool
    from envpool.registration import base_path, package_base_path

    envpool_package = Path(envpool.__file__).resolve().parent
    assets_package = Path(envpool_assets.asset_path()).resolve()

    if _is_relative_to(envpool_package, source_root):
        raise RuntimeError(
            f"imported EnvPool from source tree: {envpool_package}"
        )

    asset_version = Version(version("envpool-assets"))
    if not Version("0.1.0") <= asset_version < Version("0.2.0"):
        raise RuntimeError(
            f"unexpected envpool-assets version: {asset_version}"
        )

    if Path(base_path).resolve() != assets_package:
        raise RuntimeError(
            f"EnvPool asset base_path is {base_path}, expected {assets_package}"
        )
    if Path(package_base_path).resolve() != envpool_package:
        raise RuntimeError(
            "EnvPool package_base_path is "
            f"{package_base_path}, expected {envpool_package}"
        )

    required_assets = [
        assets_package / "atari/roms/pong.bin",
        assets_package / "gfootball/assets/data",
        assets_package / "mujoco/assets_dmc/cartpole.xml",
        assets_package / "procgen/assets/platformer/playerBlue_dead.png",
        assets_package / "vizdoom/bin/freedoom2.wad",
        assets_package / "vizdoom/maps/basic.wad",
    ]
    missing = [path for path in required_assets if not path.exists()]
    if missing:
        raise RuntimeError(f"envpool-assets missing required files: {missing}")

    required_package_files = [
        envpool_package / "vizdoom/bin/vizdoom",
        envpool_package / "vizdoom/bin/vizdoom.pk3",
    ]
    missing = [path for path in required_package_files if not path.exists()]
    if missing:
        raise RuntimeError(f"envpool wheel missing required files: {missing}")

    print(f"envpool installed at {envpool_package}")
    print(f"envpool-assets {asset_version} installed at {assets_package}")


if __name__ == "__main__":
    main()
