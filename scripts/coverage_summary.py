#!/usr/bin/env python3

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

"""Summarize and split EnvPool LCOV reports by language."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

CPP_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
}


@dataclass(frozen=True)
class LcovRecord:
    """A single LCOV source-file record."""

    source_file: str
    lines_found: int
    lines_hit: int
    branches_found: int
    branches_hit: int
    raw: str


@dataclass
class CoverageTotals:
    """Aggregate coverage totals for a group of LCOV records."""

    files: int = 0
    lines_found: int = 0
    lines_hit: int = 0
    branches_found: int = 0
    branches_hit: int = 0

    def add(self, record: LcovRecord) -> None:
        """Accumulate coverage counts from one LCOV record."""
        self.files += 1
        self.lines_found += record.lines_found
        self.lines_hit += record.lines_hit
        self.branches_found += record.branches_found
        self.branches_hit += record.branches_hit


def _classify_language(source_file: str) -> str:
    suffix = Path(source_file).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in CPP_EXTENSIONS:
        return "c_cpp"
    return "other"


def _parse_record(raw_record: str) -> LcovRecord:
    source_file = ""
    lines_found = 0
    lines_hit = 0
    branches_found = 0
    branches_hit = 0
    for line in raw_record.splitlines():
        if line.startswith("SF:"):
            source_file = line[3:]
        elif line.startswith("LF:"):
            lines_found = int(line[3:])
        elif line.startswith("LH:"):
            lines_hit = int(line[3:])
        elif line.startswith("BRF:"):
            branches_found = int(line[4:])
        elif line.startswith("BRH:"):
            branches_hit = int(line[4:])
    if not source_file:
        raise ValueError("LCOV record is missing an SF entry")
    return LcovRecord(
        source_file=source_file,
        lines_found=lines_found,
        lines_hit=lines_hit,
        branches_found=branches_found,
        branches_hit=branches_hit,
        raw=raw_record.strip() + "\n",
    )


def parse_lcov(path: Path) -> list[LcovRecord]:
    """Parse an LCOV file into per-source-file records."""
    text = path.read_text(encoding="utf-8")
    records = []
    for chunk in text.split("end_of_record"):
        chunk = chunk.strip()
        if not chunk:
            continue
        records.append(_parse_record(chunk + "\nend_of_record"))
    return records


def _format_ratio(hit: int, found: int) -> str:
    if found == 0:
        return "n/a"
    return f"{hit / found:.2%} ({hit}/{found})"


def build_summary(records: list[LcovRecord]) -> str:
    """Render a markdown summary for the provided LCOV records."""
    grouped = {
        "overall": CoverageTotals(),
        "python": CoverageTotals(),
        "c_cpp": CoverageTotals(),
        "other": CoverageTotals(),
    }
    for record in records:
        grouped["overall"].add(record)
        grouped[_classify_language(record.source_file)].add(record)

    rows = [
        ("Overall", grouped["overall"]),
        ("Python", grouped["python"]),
        ("C/C++", grouped["c_cpp"]),
    ]
    if grouped["other"].files:
        rows.append(("Other", grouped["other"]))

    lines = [
        "## Coverage Summary",
        "",
        "| Scope | Files | Line Coverage | Branch Coverage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for scope, totals in rows:
        lines.append(
            "| {scope} | {files} | {line_cov} | {branch_cov} |".format(
                scope=scope,
                files=totals.files,
                line_cov=_format_ratio(totals.lines_hit, totals.lines_found),
                branch_cov=_format_ratio(
                    totals.branches_hit, totals.branches_found
                ),
            )
        )
    return "\n".join(lines) + "\n"


def write_split_reports(records: list[LcovRecord], output_dir: Path) -> None:
    """Write per-language LCOV files into the given output directory."""
    groups = {
        "python.lcov": [
            record
            for record in records
            if _classify_language(record.source_file) == "python"
        ],
        "cpp.lcov": [
            record
            for record in records
            if _classify_language(record.source_file) == "c_cpp"
        ],
        "other.lcov": [
            record
            for record in records
            if _classify_language(record.source_file) == "other"
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, group_records in groups.items():
        if not group_records:
            continue
        (output_dir / filename).write_text(
            "".join(record.raw for record in group_records),
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lcov", type=Path, required=True, help="Path to the LCOV file."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for split LCOV reports (python/cpp/other).",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        help="Optional file path for the markdown summary.",
    )
    return parser.parse_args()


def main() -> int:
    """Summarize an LCOV report and optionally split it by language."""
    args = parse_args()
    records = parse_lcov(args.lcov)
    summary = build_summary(records)
    if args.output_dir is not None:
        write_split_reports(records, args.output_dir)
    if args.summary_file is not None:
        args.summary_file.parent.mkdir(parents=True, exist_ok=True)
        args.summary_file.write_text(summary, encoding="utf-8")
    print(summary, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
