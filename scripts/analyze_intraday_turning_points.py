#!/usr/bin/env python3
"""Run chronological QQQ research with SPY as external validation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from gex_monitor.intraday_turning_point_research import (
    ResearchConfig,
    run_feature_research,
    write_research_report,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/analysis"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis"))
    args = parser.parse_args()
    primary_path = args.input_dir / "turning_point_candidates_QQQ.parquet"
    external_path = args.input_dir / "turning_point_candidates_SPY.parquet"
    primary = pd.read_parquet(primary_path)
    external = pd.read_parquet(external_path)
    bins, rankings, bases, summary = run_feature_research(
        primary, external, ResearchConfig()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bins_path = args.output_dir / "turning_point_feature_bins_QQQ.parquet"
    ranking_path = args.output_dir / "turning_point_feature_ranking_QQQ.csv"
    bases_path = args.output_dir / "turning_point_baseline_rates_QQQ.csv"
    summary_path = args.output_dir / "turning_point_research_summary_QQQ.json"
    report_path = args.output_dir / "turning_point_research_QQQ.html"
    bins.to_parquet(bins_path, index=False)
    rankings.to_csv(ranking_path, index=False)
    bases.to_csv(bases_path, index=False)
    summary.update({
        "label_schema": "intraday_turning_point_v1",
        "primary_path": str(primary_path),
        "primary_sha256": _sha256(primary_path),
        "external_path": str(external_path),
        "external_sha256": _sha256(external_path),
        "bins_path": str(bins_path),
        "ranking_path": str(ranking_path),
        "bases_path": str(bases_path),
    })
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = write_research_report(rankings, bases, summary, report_path)
    print(json.dumps({
        "rankings": len(rankings),
        "stable_cross_symbol": summary["stable_cross_symbol"],
        "stable_qqq": summary["stable_qqq"],
        "report": str(report),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
