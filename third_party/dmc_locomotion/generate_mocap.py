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
"""Extract only the official clips used by locomotion's public factories.

This reads AST and HDF5 data, never imports or executes dm_control. Native
runtime data is little-endian float32; the smaller HDF5 files are test oracles
only and retain the original metadata and samples without transformations.
"""

import argparse
import ast
import json
import struct
from pathlib import Path

import h5py
import numpy as np


def selected_clips(source: Path) -> dict[str, list[str]]:
    """Discover the clips requested by the official public factories."""
    if source.is_file():
        source = source.parent
    subsets = ast.parse(
        (source / "locomotion/tasks/reference_pose/cmu_subsets.py").read_text()
    )
    collection = next(
        node.value
        for node in subsets.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "WALK_TINY"
            for t in node.targets
        )
    )
    assert isinstance(collection, ast.Call)
    args = {kw.arg: ast.literal_eval(kw.value) for kw in collection.keywords}
    if set(args) != {"ids"}:
        raise ValueError("WALK_TINY now specifies clip ranges or weights")
    initializer = ast.parse(
        (source / "locomotion/walkers/initializers/mocap.py").read_text()
    )
    cls = next(
        n
        for n in initializer.body
        if isinstance(n, ast.ClassDef) and n.name == "CMUMocapInitializer"
    )
    init = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    defaults = {
        arg.arg: ast.literal_eval(value)
        for arg, value in zip(
            init.args.args[-len(init.args.defaults) :],
            init.args.defaults,
            strict=False,
        )
    }
    if defaults["version"] != "2019":
        raise ValueError("Soccer's default mocap version changed")
    return {"2019": [defaults["mocap_key"]], "2020": list(args["ids"])}


def generate(
    source: Path, cmu_2019: Path, cmu_2020: Path, output: Path
) -> None:
    """Write native samples and unchanged, reduced test-only HDF5 archives."""
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for version, keys in selected_clips(source).items():
        path = cmu_2019 if version == "2019" else cmu_2020
        manifest[version] = []
        with (
            h5py.File(path, "r") as upstream,
            h5py.File(output / f"oracle_{version}.h5", "w") as oracle,
            (output / f"mocap_{version}.bin").open("wb") as binary,
        ):
            binary.write(b"EPMOCAP1")
            binary.write(struct.pack("<I", len(keys)))

            def write_string(value: str) -> None:
                encoded = value.encode("utf-8")
                binary.write(struct.pack("<I", len(encoded)))
                binary.write(encoded)

            for key in keys:
                group = upstream[key]
                upstream.copy(group, oracle, name=key)
                frames = int(group.attrs["num_steps"])
                dt = float(group.attrs["dt"])
                walker = group["walkers/walker_0"]
                features = {
                    name: value
                    for name, value in walker.items()
                    if isinstance(value, h5py.Dataset) and value.ndim == 2
                }
                write_string(key)
                binary.write(struct.pack("<IdI", frames, dt, len(features)))
                for name, feature in features.items():
                    if feature.shape[1] != frames or feature.dtype != np.dtype(
                        "float32"
                    ):
                        raise ValueError(
                            f"Unexpected mocap feature format: {key}/{name}"
                        )
                    write_string(name)
                    binary.write(struct.pack("<I", feature.shape[0]))
                    binary.write(
                        np.asarray(feature[:].T, dtype="<f4").tobytes()
                    )
                manifest[version].append({
                    "id": key,
                    "frames": frames,
                    "dt": dt,
                })
    (output / "mocap.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cmu-2019", type=Path, required=True)
    parser.add_argument("--cmu-2020", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.cmu_2019, args.cmu_2020, args.output)
