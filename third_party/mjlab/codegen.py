# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Move portable generated inputs between build hosts, never into a wheel.

The official ARM Warp SDK requires a newer glibc than manylinux 2.28. Generate
on a supported host, then compile the exported C++ in the release container.
The archive carries recipe and payload hashes so stale inputs fail the build.
"""

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath


def digest(data: bytes) -> str:
    """Hash a recipe or generated file."""
    return hashlib.sha256(data).hexdigest()


def pack(archive: Path, payload: list[list[str]], recipe: dict) -> None:
    """Write only declared generated outputs with deterministic tar metadata."""
    files = {}
    for name, source in payload:
        path = Path(source)
        if path.is_dir():
            files.update({
                str(
                    PurePosixPath(name) / child.relative_to(path).as_posix()
                ): child
                for child in path.rglob("*")
                if child.is_file()
            })
        else:
            files[name] = path
    manifest = {
        "format": 1,
        "recipe": recipe,
        "files": {
            name: digest(path.read_bytes()) for name, path in files.items()
        },
    }
    archive.parent.mkdir(parents=True, exist_ok=True)
    with (
        archive.open("wb") as output,
        gzip.GzipFile(
            fileobj=output, mode="wb", mtime=0, filename=""
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as bundle,
    ):
        for name in ["manifest.json", *sorted(files)]:
            data = (
                json.dumps(manifest, sort_keys=True).encode()
                if name == "manifest.json"
                else files[name].read_bytes()
            )
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), 0o644
            bundle.addfile(member, io.BytesIO(data))


def unpack(archive: Path, output: Path, recipe: dict) -> None:
    """Validate the recipe, paths and payload before exposing build inputs."""
    with tarfile.open(archive, "r:gz") as bundle:
        source = bundle.extractfile("manifest.json")
        if source is None:
            raise ValueError("MJLab codegen manifest is missing")
        manifest = json.load(source)
        if manifest.get("format") != 1 or manifest.get("recipe") != recipe:
            raise ValueError(
                "stale MJLab codegen input; regenerate codegen_bundle"
            )
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(set(names)) != len(names) or set(names) != {
            "manifest.json",
            *manifest["files"],
        }:
            raise ValueError("MJLab codegen archive inventory mismatch")
        for member in members:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or "\\" in member.name
                or ":" in member.name
                or ".." in path.parts
                or str(path) != member.name
                or (
                    member.name != "manifest.json"
                    and path.parts[0] not in {"generated", "warp_headers"}
                )
            ):
                raise ValueError(f"unsafe MJLab codegen member: {member.name}")
            if member.name == "manifest.json":
                continue
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"missing MJLab codegen member: {member.name}")
            data = source.read()
            if digest(data) != manifest["files"][member.name]:
                raise ValueError(f"corrupt MJLab codegen member: {member.name}")
            destination = output / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)


def main() -> None:
    """Pack or restore the exact outputs declared by the Bazel rule."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pack", "unpack"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--recipe", nargs=2, action="append", required=True)
    parser.add_argument("--payload", nargs=2, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    recipe = {
        name: digest(Path(path).read_bytes()) for name, path in args.recipe
    }
    if args.mode == "pack":
        pack(args.archive, args.payload, recipe)
    else:
        if args.output is None:
            parser.error("unpack requires --output")
        unpack(args.archive, args.output, recipe)


if __name__ == "__main__":
    main()
