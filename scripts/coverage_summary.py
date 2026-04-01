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
import html
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

    @property
    def line_ratio(self) -> float | None:
        """Return the line coverage ratio, if available."""
        if self.lines_found == 0:
            return None
        return self.lines_hit / self.lines_found


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


def _build_grouped_totals(
    records: list[LcovRecord],
) -> dict[str, CoverageTotals]:
    grouped = {
        "overall": CoverageTotals(),
        "python": CoverageTotals(),
        "c_cpp": CoverageTotals(),
        "other": CoverageTotals(),
    }
    for record in records:
        grouped["overall"].add(record)
        grouped[_classify_language(record.source_file)].add(record)
    return grouped


def build_summary(records: list[LcovRecord]) -> str:
    """Render a markdown summary for the provided LCOV records."""
    grouped = _build_grouped_totals(records)

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


def _badge_color(line_ratio: float | None) -> str:
    if line_ratio is None:
        return "#9f9f9f"
    if line_ratio >= 0.9:
        return "#4c1"
    if line_ratio >= 0.75:
        return "#97CA00"
    if line_ratio >= 0.6:
        return "#dfb317"
    if line_ratio >= 0.4:
        return "#fe7d37"
    return "#e05d44"


def _badge_text_width(text: str) -> int:
    return max(40, 8 * len(text) + 10)


def build_badge(records: list[LcovRecord], label: str = "coverage") -> str:
    """Render a static SVG badge for overall line coverage."""
    overall = _build_grouped_totals(records)["overall"]
    line_ratio = overall.line_ratio
    message = "n/a" if line_ratio is None else f"{line_ratio * 100:.1f}%"
    label_width = _badge_text_width(label)
    message_width = _badge_text_width(message)
    total_width = label_width + message_width
    label_x = label_width / 2
    message_x = label_width + message_width / 2
    color = _badge_color(line_ratio)
    escaped_label = html.escape(label)
    escaped_message = html.escape(message)
    aria_label = html.escape(f"{label}: {message}")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20" role="img" aria-label="{aria_label}">
  <title>{aria_label}</title>
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#fff" stop-opacity=".7"/>
    <stop offset=".1" stop-color="#aaa" stop-opacity=".1"/>
    <stop offset=".9" stop-opacity=".3"/>
    <stop offset="1" stop-opacity=".5"/>
  </linearGradient>
  <mask id="a">
    <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
  </mask>
  <g mask="url(#a)">
    <rect width="{label_width}" height="20" fill="#555"/>
    <rect x="{label_width}" width="{message_width}" height="20" fill="{color}"/>
    <rect width="{total_width}" height="20" fill="url(#b)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
    <text x="{label_x}" y="15" fill="#010101" fill-opacity=".3">{escaped_label}</text>
    <text x="{label_x}" y="14">{escaped_label}</text>
    <text x="{message_x}" y="15" fill="#010101" fill-opacity=".3">{escaped_message}</text>
    <text x="{message_x}" y="14">{escaped_message}</text>
  </g>
</svg>
"""


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
    parser.add_argument(
        "--badge-file",
        type=Path,
        help="Optional file path for a static SVG badge.",
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
    if args.badge_file is not None:
        args.badge_file.parent.mkdir(parents=True, exist_ok=True)
        args.badge_file.write_text(build_badge(records), encoding="utf-8")
    print(summary, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
