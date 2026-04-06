#!/usr/bin/env python3

"""Optimize built wheels before publish.

This post-processes wheels to keep the final file size down:

1. Strip native binaries when a platform-appropriate strip tool is available.
2. Repack the wheel with a fresh `RECORD` using max zip compression.

Dead Gymnasium-Robotics assets are now pruned at the Bazel `third_party`
target layer, before they can enter the wheel.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

_ELF_MAGIC = b"\x7fELF"
_PE_MAGIC = b"MZ"
_MACHO_MAGICS = {
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="+", help="Wheel file(s) to optimize.")
    return parser.parse_args()


def _unpack_wheel(wheel_path: Path, unpack_dir: Path) -> None:
    with ZipFile(wheel_path) as zf:
        zf.extractall(unpack_dir)


def _find_dist_info_dir(unpack_dir: Path) -> Path:
    matches = sorted(unpack_dir.glob("*.dist-info"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one .dist-info dir in {unpack_dir}, found {matches}"
        )
    return matches[0]


def _binary_kind(path: Path) -> str | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            header = f.read(4)
    except OSError:
        return None
    if header == _ELF_MAGIC:
        return "elf"
    if header[:2] == _PE_MAGIC:
        return "pe"
    if header in _MACHO_MAGICS:
        return "macho"
    return None


def _strip_command(kind: str) -> list[str] | None:
    if kind == "elf":
        strip_bin = shutil.which("strip")
        if strip_bin:
            return [strip_bin, "--strip-unneeded"]
        return None
    if kind == "macho":
        strip_bin = shutil.which("strip")
        if strip_bin:
            return [strip_bin, "-x"]
        return None
    if kind == "pe":
        llvm_strip = shutil.which("llvm-strip") or shutil.which(
            "llvm-strip.exe"
        )
        if llvm_strip:
            return [llvm_strip, "-s"]
        strip_bin = shutil.which("strip")
        if strip_bin:
            return [strip_bin, "--strip-unneeded"]
        return None
    raise ValueError(f"unsupported binary kind: {kind}")


def _strip_native_binaries(unpack_dir: Path) -> tuple[int, list[str]]:
    stripped = 0
    failures: list[str] = []
    for path in sorted(unpack_dir.rglob("*")):
        kind = _binary_kind(path)
        if kind is None:
            continue
        command = _strip_command(kind)
        if command is None:
            failures.append(f"{path}: no strip tool available for {kind}")
            continue
        result = subprocess.run(
            [*command, str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            stripped += 1
            continue
        failures.append(f"{path}: {result.stderr.strip()}")
    return stripped, failures


def _record_row(rel_path: str, data: bytes) -> list[str]:
    digest = hashlib.sha256(data).digest()
    b64 = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return [rel_path, f"sha256={b64}", str(len(data))]


def _write_wheel(unpack_dir: Path, wheel_path: Path) -> None:
    dist_info_dir = _find_dist_info_dir(unpack_dir)
    record_rel = f"{dist_info_dir.name}/RECORD"
    record_rows: list[list[str]] = []

    with tempfile.NamedTemporaryFile(
        prefix=wheel_path.stem + ".",
        suffix=".whl",
        dir=wheel_path.parent,
        delete=False,
    ) as tmp_file:
        tmp_wheel = Path(tmp_file.name)

    try:
        with ZipFile(
            tmp_wheel, "w", compression=ZIP_DEFLATED, compresslevel=9
        ) as zf:
            for path in sorted(unpack_dir.rglob("*")):
                if not path.is_file():
                    continue
                rel_path = path.relative_to(unpack_dir).as_posix()
                if rel_path == record_rel:
                    continue
                data = path.read_bytes()
                zf.writestr(rel_path, data)
                record_rows.append(_record_row(rel_path, data))

            record_rows.sort(key=lambda row: row[0])
            record_rows.append([record_rel, "", ""])
            record_content = "".join(
                ",".join(row) + "\n" for row in record_rows
            )
            zf.writestr(record_rel, record_content.encode("utf-8"))

        tmp_wheel.replace(wheel_path)
    finally:
        tmp_wheel.unlink(missing_ok=True)


def _format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MiB"


def _optimize_wheel(wheel_path: Path) -> None:
    if not wheel_path.is_file():
        raise FileNotFoundError(f"Wheel not found: {wheel_path}")

    before_size = wheel_path.stat().st_size
    with tempfile.TemporaryDirectory(prefix=wheel_path.stem + ".") as tmp_dir:
        unpack_dir = Path(tmp_dir) / "wheel"
        _unpack_wheel(wheel_path, unpack_dir)
        stripped_binaries, failures = _strip_native_binaries(unpack_dir)
        if failures:
            details = "; ".join(failures)
            raise RuntimeError(
                f"failed to strip native binaries in {wheel_path.name}: {details}"
            )
        _write_wheel(unpack_dir, wheel_path)

    after_size = wheel_path.stat().st_size
    delta = before_size - after_size
    print(
        f"{wheel_path.name}: stripped {stripped_binaries} native files, "
        f"saved {delta} bytes ({_format_bytes(delta)}), "
        f"final size {after_size} bytes ({_format_bytes(after_size)})"
    )


def main() -> int:
    """Optimize each wheel passed on the command line."""
    args = _parse_args()
    for wheel in args.wheels:
        _optimize_wheel(Path(wheel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
