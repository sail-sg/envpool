#!/usr/bin/env python3

"""Optimize repaired Linux wheels before publish.

This post-processes `auditwheel repair` outputs to keep the final wheel below
PyPI's file-size limit:

1. Strip ELF binaries to shrink both EnvPool extensions and vendored
   `auditwheel` dependencies.
2. Repack the wheel with a fresh `RECORD`.

Dead Gymnasium-Robotics assets are now pruned at the Bazel `third_party`
target layer, before they can enter the wheel.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

_ELF_MAGIC = b"\x7fELF"


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


def _is_elf_binary(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        with path.open("rb") as f:
            return f.read(4) == _ELF_MAGIC
    except OSError:
        return False


def _strip_elf_binaries(unpack_dir: Path) -> tuple[int, list[str]]:
    stripped = 0
    failures: list[str] = []
    for path in sorted(unpack_dir.rglob("*")):
        if not _is_elf_binary(path):
            continue
        # `strip --strip-unneeded` works for shared objects and executables.
        result = subprocess.run(
            ["strip", "--strip-unneeded", str(path)],
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
        prefix=wheel_path.stem + ".", suffix=".whl", delete=False
    ) as tmp_file:
        tmp_wheel = Path(tmp_file.name)

    try:
        with ZipFile(tmp_wheel, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
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
            record_content = "".join(",".join(row) + "\n" for row in record_rows)
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
        stripped_binaries, failures = _strip_elf_binaries(unpack_dir)
        _write_wheel(unpack_dir, wheel_path)

    after_size = wheel_path.stat().st_size
    delta = before_size - after_size
    print(
        f"{wheel_path.name}: stripped {stripped_binaries} ELF files, "
        f"saved {delta} bytes ({_format_bytes(delta)}), "
        f"final size {after_size} bytes ({_format_bytes(after_size)})"
    )
    for failure in failures:
        print(f"warning: strip skipped for {failure}", file=sys.stderr)


def main() -> int:
    """Optimize each wheel passed on the command line."""
    args = _parse_args()
    for wheel in args.wheels:
        _optimize_wheel(Path(wheel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
