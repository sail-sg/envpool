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
"""Losslessly share serialized model/graph arrays across MJLab task presets."""

import bisect
import hashlib
import itertools
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import zstandard


def array_parts(model: object, name: str, value: np.ndarray) -> Iterable[bytes]:
    """Use mesh boundaries: adding terrain shifts the body's BVH prefix."""
    index = (
        "mesh_bvhadr"
        if name.startswith("bvh_")
        else "mesh_vertadr"
        if name == "mesh_vert"
        else "mesh_normaladr"
        if name == "mesh_normal"
        else "mesh_faceadr"
        if name in ("mesh_face", "mesh_facenormal", "mesh_facetexcoord")
        else None
    )
    if index is None or value.ndim == 0:
        yield value.tobytes()
        return
    bounds = sorted({
        0,
        len(value),
        *(int(x) for x in getattr(model, index) if 0 <= x < len(value)),
    })
    for start, end in itertools.pairwise(bounds):
        yield value[start:end].tobytes()


def prune_assets(root: Path) -> None:
    """Discard obsolete blobs after regenerating an existing export directory."""
    needed = set()
    for path in root.glob("*/*.*.json"):
        index = json.loads(path.read_text())
        needed.add(index["remainder"])
        needed.update(piece["blob"] for piece in index["pieces"])
    for path in (root / "shared").glob("*.zst"):
        if path.stem not in needed:
            path.unlink()


def pack_binary(path: Path, arrays: Iterable[bytes]) -> None:
    """Store repeated arrays once; preserve every remaining serialized byte.

    MuJoCo's render-only host models include large mesh BVHs, often identical
    across robot tasks. Warp graphs repeat some of those geometry arrays too.
    Exact byte slices are factored out without changing either file format or
    dropping collision/render data. Reconstruction is checked before removal.
    """
    original = path.read_bytes()
    remainder = bytearray(original)
    shared = path.parent.parent / "shared"
    shared.mkdir(exist_ok=True)
    compressor = zstandard.ZstdCompressor(level=15, write_checksum=True)
    decompressor = zstandard.ZstdDecompressor()

    def store(data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        output = shared / (digest + ".zst")
        if not output.exists():
            encoded = compressor.compress(data)
            if decompressor.decompress(encoded) != data:
                raise ValueError("asset compression changed serialized bytes")
            output.write_bytes(encoded)
        return digest

    starts: list[int] = []
    pieces: list[dict] = []
    # Process the original array boundaries, not arbitrary file blocks; the
    # same geometry can occur at different offsets in different task models.
    candidates = {
        data for data in arrays if len(data) >= 4096 and data.strip(b"\0")
    }
    for data in sorted(candidates, key=lambda value: (-len(value), value)):
        offset = 0
        while (offset := original.find(data, offset)) >= 0:
            position = bisect.bisect_left(starts, offset)
            end = offset + len(data)
            overlaps = (
                position > 0
                and pieces[position - 1]["offset"]
                + pieces[position - 1]["size"]
                > offset
            ) or (position < len(starts) and starts[position] < end)
            if not overlaps:
                starts.insert(position, offset)
                pieces.insert(
                    position,
                    {"offset": offset, "size": len(data), "blob": store(data)},
                )
                remainder[offset:end] = bytes(len(data))
            offset = end
    index = {
        "size": len(original),
        "remainder": store(bytes(remainder)),
        "pieces": pieces,
    }
    restored = bytearray(
        decompressor.decompress(
            (shared / (index["remainder"] + ".zst")).read_bytes()
        )
    )
    for piece in pieces:
        data = decompressor.decompress(
            (shared / (piece["blob"] + ".zst")).read_bytes()
        )
        restored[piece["offset"] : piece["offset"] + piece["size"]] = data
    if restored != original:
        raise ValueError(f"asset factoring changed serialized bytes: {path}")
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(index, separators=(",", ":")) + "\n"
    )
    path.unlink()
