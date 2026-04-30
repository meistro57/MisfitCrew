#!/usr/bin/env python3
"""
misfit_report_pull.py

Pull all misfit_reports for a given source_file out of Qdrant and write them
to a clean, readable Markdown file. Intended for reviewing ROOT ACCESS (or any
book) against the critical-analysis pipeline's flagged issues before release.

Usage:
    python misfit_report_pull.py
        -> defaults to ROOT-ACCESS book, writes to ./reviews/ROOT-ACCESS_review.md

    python misfit_report_pull.py --source Bashar-Channeled-by-Darryl-Anka-Complete-Transcripts.pdf
        -> pulls Bashar reports instead

    python misfit_report_pull.py --list
        -> lists all source_files present in misfit_reports with counts

    python misfit_report_pull.py --all
        -> writes one markdown file per source_file into ./reviews/

Requires:
    pip install qdrant-client

Environment:
    QDRANT_URL  (default: http://localhost:6333)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


# ---------------------------------------------------------------- config ----

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "misfit_reports"
DEFAULT_SOURCE = "ROOT-ACCESS_A-Misfits-Complete-Guide-to-Reality-Engineering.md"
OUTPUT_DIR = Path("./ROOTreviews")
SCROLL_BATCH = 100


# ----------------------------------------------------------------- util -----

def parse_mined_at(s: str) -> datetime:
    """Parse ISO-ish timestamps; tolerate trailing microseconds or Z suffixes."""
    if not s:
        return datetime.min
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        # last-ditch: strip microseconds
        try:
            return datetime.fromisoformat(s.split(".")[0])
        except Exception:
            return datetime.min


def safe_filename(s: str) -> str:
    """Turn a source_file into something filesystem-safe."""
    stem = s.rsplit(".", 1)[0]
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)


# -------------------------------------------------------------- loaders -----

def list_source_files(client: QdrantClient) -> dict[str, int]:
    """Scroll the whole collection, count points per source_file."""
    counts: dict[str, int] = defaultdict(int)
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=SCROLL_BATCH,
            with_payload=["source_file"],
            with_vectors=False,
            offset=offset,
        )
        for p in points:
            sf = (p.payload or {}).get("source_file", "<unknown>")
            counts[sf] += 1
        if offset is None:
            break
    return dict(counts)


def load_reports_for_source(
    client: QdrantClient, source_file: str
) -> list[dict]:
    """Pull every misfit_report for a given source_file, full payloads."""
    reports: list[dict] = []
    offset = None
    flt = qm.Filter(
        must=[
            qm.FieldCondition(
                key="source_file",
                match=qm.MatchValue(value=source_file),
            )
        ]
    )
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=SCROLL_BATCH,
            scroll_filter=flt,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for p in points:
            payload = p.payload or {}
            reports.append({
                "id": str(p.id),
                "mined_at": payload.get("mined_at", ""),
                "reasoning": payload.get("reasoning", ""),
                "report": payload.get("report", ""),
                "verdict": payload.get("verdict", ""),
                "source_file": payload.get("source_file", source_file),
            })
        if offset is None:
            break
    reports.sort(key=lambda r: parse_mined_at(r["mined_at"]))
    return reports


# -------------------------------------------------------------- writer ------

def format_timestamp(s: str) -> str:
    dt = parse_mined_at(s)
    if dt == datetime.min:
        return s or "unknown"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def render_markdown(source_file: str, reports: list[dict]) -> str:
    """Render one markdown document from a list of reports."""
    lines: list[str] = []
    title = f"Critical Review: {source_file}"
    lines.append(f"# {title}")
    lines.append("")
    lines.append(
        f"*Compiled from the `misfit_reports` pipeline on "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}.*"
    )
    lines.append("")
    lines.append(f"**Total reports:** {len(reports)}")
    lines.append("")
    lines.append(
        "Each report was produced by an autonomous critical-analysis pass "
        "looking for **Hardware Glitches** (logical inconsistencies, factual "
        "errors, or conceptual breakdowns) and **Ontological Shock** "
        "(paradigm-disruptive claims that deserve extra scrutiny). A second "
        "pass then renders a **Verdict** that evaluates the report's own "
        "internal logical consistency and its validity within the 'Reality "
        "Engineering' domain frame."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # TOC
    lines.append("## Table of Contents")
    lines.append("")
    for i, r in enumerate(reports, 1):
        ts = format_timestamp(r["mined_at"])
        lines.append(f"{i}. [Report {i} — {ts}](#report-{i})")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Reports
    for i, r in enumerate(reports, 1):
        ts = format_timestamp(r["mined_at"])
        lines.append(f"## Report {i}")
        lines.append("")
        lines.append(f"- **Point ID:** `{r['id']}`")
        lines.append(f"- **Mined at:** {ts}")
        lines.append("")

        if r["reasoning"]:
            lines.append("### Reasoning (scratch work)")
            lines.append("")
            lines.append(r["reasoning"].strip())
            lines.append("")

        if r["report"]:
            lines.append("### Report")
            lines.append("")
            lines.append(r["report"].strip())
            lines.append("")

        if r["verdict"]:
            lines.append("### Verdict (meta-critique of the report)")
            lines.append("")
            lines.append(r["verdict"].strip())
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def write_review(source_file: str, reports: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_filename(source_file)
    path = out_dir / f"{stem}_review.md"
    text = render_markdown(source_file, reports)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------- main ------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=f"source_file to pull (default: {DEFAULT_SOURCE})",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="list all source_files present in misfit_reports and exit",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="write one review file per source_file",
    )
    ap.add_argument(
        "--out",
        default=str(OUTPUT_DIR),
        help=f"output directory (default: {OUTPUT_DIR})",
    )
    args = ap.parse_args()

    client = QdrantClient(url=QDRANT_URL)
    out_dir = Path(args.out)

    if args.list:
        counts = list_source_files(client)
        if not counts:
            print("No reports found in misfit_reports.")
            return 0
        width = max(len(sf) for sf in counts)
        print(f"{'source_file'.ljust(width)}  count")
        print(f"{'-' * width}  -----")
        for sf, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"{sf.ljust(width)}  {n:>5}")
        print(f"\nTotal source files: {len(counts)}")
        print(f"Total reports:      {sum(counts.values())}")
        return 0

    if args.all:
        counts = list_source_files(client)
        if not counts:
            print("No reports found in misfit_reports.")
            return 0
        print(f"Writing {len(counts)} review files to {out_dir}/ ...")
        for sf in sorted(counts):
            reports = load_reports_for_source(client, sf)
            if not reports:
                continue
            path = write_review(sf, reports, out_dir)
            print(f"  {sf}  ->  {path}  ({len(reports)} reports)")
        return 0

    # single source mode
    reports = load_reports_for_source(client, args.source)
    if not reports:
        print(
            f"No reports found for source_file={args.source!r}.\n"
            f"Try --list to see what is available."
        )
        return 1
    path = write_review(args.source, reports, out_dir)
    print(f"Wrote {len(reports)} reports to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
