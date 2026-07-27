#!/usr/bin/env python3
"""Build offline intraday turning-point datasets and an SVG audit gallery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gex_monitor.intraday_turning_points import (
    TurningPointConfig,
    build_symbol_dataset,
    select_audit_events,
    write_audit_gallery,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="QQQ", choices=["QQQ", "SPY"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis"))
    parser.add_argument("--audit-count", type=int, default=50)
    parser.add_argument(
        "--gex-method", choices=["stored", "oi_position_v2"], default="stored",
    )
    args = parser.parse_args()

    config = TurningPointConfig()
    minutes, events, day_frames, summary = build_symbol_dataset(
        args.data_dir, args.symbol, config, gex_method=args.gex_method
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    minute_path = args.output_dir / f"turning_point_minutes_{args.symbol}.parquet"
    candidate_path = args.output_dir / (
        f"turning_point_candidates_{args.symbol}.parquet"
    )
    event_path = args.output_dir / f"turning_points_{args.symbol}.parquet"
    summary_path = args.output_dir / f"turning_point_summary_{args.symbol}.json"
    minutes.to_parquet(minute_path, index=False)
    events.to_parquet(candidate_path, index=False)
    turning_points = events[events["outcome"].isin([
        "reversal_up", "reversal_down"
    ])].copy()
    turning_points.to_parquet(event_path, index=False)
    selected = select_audit_events(events, args.audit_count)
    summary.update({
        "minute_path": str(minute_path),
        "candidate_path": str(candidate_path),
        "event_path": str(event_path),
        "audit_events": len(selected),
    })
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = write_audit_gallery(selected, day_frames, args.output_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"audit_report={report}")


if __name__ == "__main__":
    main()
