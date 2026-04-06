#!/usr/bin/env python3
"""Check that one or more wheel files stay below a byte-size limit."""

from __future__ import annotations

import argparse
import glob
from collections import OrderedDict
from pathlib import Path


def _resolve_wheels(targets: list[str]) -> list[Path]:
    wheels: OrderedDict[Path, None] = OrderedDict()
    missing: list[str] = []
    invalid: list[Path] = []

    for target in targets:
        matches = [Path(match) for match in glob.glob(target)]
        if not matches:
            candidate = Path(target)
            if candidate.exists():
                matches = [candidate]
            else:
                missing.append(target)
                continue
        for match in matches:
            if match.is_dir():
                for wheel in sorted(match.glob("*.whl")):
                    wheels[wheel.resolve()] = None
                continue
            if match.suffix != ".whl":
                invalid.append(match)
                continue
            wheels[match.resolve()] = None

    if missing:
        missing_targets = ", ".join(sorted(missing))
        raise SystemExit(f"wheel input not found: {missing_targets}")
    if invalid:
        invalid_targets = ", ".join(str(path) for path in sorted(invalid))
        raise SystemExit(
            f"non-wheel inputs are not supported: {invalid_targets}"
        )
    if not wheels:
        raise SystemExit("no wheel files found to size-check")
    return list(wheels)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "targets", nargs="+", help="wheel files, directories, or glob patterns"
    )
    parser.add_argument(
        "--limit-bytes",
        type=int,
        required=True,
        help="maximum allowed size in bytes",
    )
    return parser.parse_args()


def main() -> None:
    """Validate that all requested wheel files are below the configured limit."""
    args = _parse_args()
    limit = args.limit_bytes
    limit_mib = limit / (1024 * 1024)
    oversize: list[tuple[Path, int]] = []

    for wheel in _resolve_wheels(args.targets):
        size = wheel.stat().st_size
        size_mib = size / (1024 * 1024)
        print(f"{wheel.name}: {size} bytes ({size_mib:.2f} MiB)")
        if size >= limit:
            oversize.append((wheel, size))

    if oversize:
        failures = ", ".join(f"{wheel.name}={size}" for wheel, size in oversize)
        raise SystemExit(
            f"wheel size check failed: {failures} bytes exceed limit {limit} ({limit_mib:.2f} MiB)"
        )

    print(
        f"wheel size check passed: all wheels are below {limit} bytes ({limit_mib:.2f} MiB)"
    )


if __name__ == "__main__":
    main()
