#!/usr/bin/env python3
"""Run evaluation_accident analysis pipeline for results_grid_accident_multi in one command."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


ACCIDENT_DIR_RE = re.compile(r"^accident_(\d+)_pf_(\d+)$")


@dataclass
class AccidentTarget:
    tag: str
    frame: int
    path: Path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--multi-root",
        type=Path,
        default=Path("results_grid_accident_multi"),
        help="Root directory that contains accident_*_pf_* directories",
    )
    p.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("exp_future/collisions_exp_accident.csv"),
        help="Baseline collision CSV used by predicted risk scripts",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root directory (default: <multi-root>/analysis_multi)",
    )
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--threshold-m", type=float, default=5.0)
    p.add_argument("--window-from-switch-sec", type=float, default=10.0)
    p.add_argument(
        "--risk-mode",
        choices=("speed", "pair"),
        default="speed",
        help="Choose predicted risk script variant",
    )
    p.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional method filter passed to hotspot summarizer (e.g. autopilot lstm)",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on first accident directory failure",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def discover_targets(multi_root: Path) -> List[AccidentTarget]:
    targets: List[AccidentTarget] = []
    for p in sorted(multi_root.iterdir()):
        if not p.is_dir():
            continue
        m = ACCIDENT_DIR_RE.match(p.name)
        if not m:
            continue
        frame = int(m.group(2))
        targets.append(AccidentTarget(tag=p.name, frame=frame, path=p))
    return targets


def run_cmd(args: Sequence[str], cwd: Path) -> None:
    subprocess.run(list(args), cwd=str(cwd), check=True)


def aggregate_risk_global(per_run: pd.DataFrame) -> pd.DataFrame:
    if per_run.empty:
        return per_run

    required = {"method", "lead_sec", "risk_flag"}
    if not required.issubset(per_run.columns):
        return pd.DataFrame()

    keys = ["method", "lead_sec"]
    grouped = per_run.groupby(keys, as_index=False)
    out = grouped.agg(
        n_runs_total=("risk_flag", "size"),
        risk_rate=("risk_flag", "mean"),
    )

    optional_mean_cols = [
        "n_frames_in_window",
        "n_frames_inside_region",
        "max_consecutive_inside_frames",
        "min_dist_from_baseline",
        "n_entered_actors",
    ]
    for col in optional_mean_cols:
        if col in per_run.columns:
            mean_df = (
                per_run.groupby(keys)[col]
                .mean()
                .reset_index(name=f"avg_{col}")
            )
            out = out.merge(mean_df, on=keys, how="left")

    return out.sort_values(["method", "lead_sec"]).reset_index(drop=True)


def aggregate_collision_global(per_run: pd.DataFrame) -> pd.DataFrame:
    if per_run.empty:
        return per_run

    required = {"method", "lead_sec"}
    if not required.issubset(per_run.columns):
        return pd.DataFrame()

    keys = ["method", "lead_sec"]
    out = (
        per_run.groupby(keys, as_index=False)
        .size()
        .rename(columns={"size": "n_runs_total"})
    )

    numeric_cols = [
        c
        for c in per_run.columns
        if c not in {"accident_tag", "accident_frame", "method", "lead_sec", "rep"}
        and pd.api.types.is_numeric_dtype(per_run[c])
    ]

    for col in numeric_cols:
        sum_df = (
            per_run.groupby(keys)[col]
            .sum()
            .reset_index(name=f"{col}_sum")
        )
        mean_df = (
            per_run.groupby(keys)[col]
            .mean()
            .reset_index(name=f"{col}_mean")
        )
        out = out.merge(sum_df, on=keys, how="left")
        out = out.merge(mean_df, on=keys, how="left")

    return out.sort_values(["method", "lead_sec"]).reset_index(drop=True)


def aggregate_per_method_like_global(per_method_all: pd.DataFrame) -> pd.DataFrame:
    if per_method_all.empty:
        return per_method_all

    keys = ["method", "lead_sec"]
    if not set(keys).issubset(per_method_all.columns):
        return pd.DataFrame()

    numeric_cols = [
        c
        for c in per_method_all.columns
        if c not in {"accident_tag", "accident_frame", "method", "lead_sec"}
        and pd.api.types.is_numeric_dtype(per_method_all[c])
    ]
    if not numeric_cols:
        return per_method_all[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)

    out = (
        per_method_all.groupby(keys, as_index=False)[numeric_cols]
        .sum()
        .sort_values(keys)
        .reset_index(drop=True)
    )
    return out


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    eval_dir = repo_root / "evaluation_accident"

    multi_root = args.multi_root.expanduser()
    if not multi_root.is_absolute():
        multi_root = (repo_root / multi_root).resolve()

    baseline_csv = args.baseline_csv.expanduser()
    if not baseline_csv.is_absolute():
        baseline_csv = (repo_root / baseline_csv).resolve()

    if args.out_root is None:
        out_root = (multi_root / "analysis_multi").resolve()
    else:
        out_root = args.out_root.expanduser()
        if not out_root.is_absolute():
            out_root = (repo_root / out_root).resolve()

    if not multi_root.exists():
        raise FileNotFoundError(f"multi-root not found: {multi_root}")
    if not baseline_csv.exists():
        raise FileNotFoundError(f"baseline csv not found: {baseline_csv}")

    out_root.mkdir(parents=True, exist_ok=True)

    targets = discover_targets(multi_root)
    if not targets:
        raise RuntimeError(f"No accident_*_pf_* directories found under {multi_root}")

    index_rows = []
    combined_risk_rows = []
    combined_coll_rows = []
    combined_risk_per_run_rows = []
    combined_coll_per_run_rows = []

    for t in targets:
        accident_out = out_root / t.tag
        events_out = accident_out / "events"
        mean_plot_out = accident_out / "fig_mean_events"
        scatter_out = accident_out / "fig_scatter"

        accident_out.mkdir(parents=True, exist_ok=True)

        try:
            # 1) events extraction
            cmd1 = [
                sys.executable,
                str(eval_dir / "summarize_hotspots_with_nearmiss.py"),
                "--root",
                str(t.path),
                "--outdir",
                str(events_out),
                "--base-payload-frame",
                str(t.frame),
                "--dt",
                str(args.dt),
                "--threshold-m",
                str(args.threshold_m),
                "--include-nearmiss",
                "--include-collisions",
            ]
            if args.methods:
                cmd1 += ["--methods", *args.methods]
            run_cmd(cmd1, repo_root)

            # 2) mean events plot
            run_cmd(
                [
                    sys.executable,
                    str(eval_dir / "plot_mean_events_by_lead.py"),
                    "--events-dir",
                    str(events_out),
                    "--outdir",
                    str(mean_plot_out),
                ],
                repo_root,
            )

            # 3) scatter plots
            run_cmd(
                [
                    sys.executable,
                    str(eval_dir / "plot_events_scatter_pdf.py"),
                    "--events-dir",
                    str(events_out),
                    "--outdir",
                    str(scatter_out),
                    "--make-merged-methods",
                    "--merged-target",
                    "all",
                ],
                repo_root,
            )

            # 4) near miss + collisions summary
            per_run_csv = accident_out / "near_miss_and_collisions_per_run.csv"
            per_method_csv = accident_out / "near_miss_and_collisions_per_method.csv"
            run_cmd(
                [
                    sys.executable,
                    str(eval_dir / "summarize_near_miss_and_collisions_from_dirs.py"),
                    "--base-dir",
                    str(t.path),
                    "--near-miss-events",
                    str(events_out / "events_nearmiss.csv"),
                    "--out-per-run",
                    str(per_run_csv),
                    "--out-per-method",
                    str(per_method_csv),
                ],
                repo_root,
            )

            # 5) predicted risk
            risk_per_run_csv = accident_out / "predicted_risk_per_run.csv"
            risk_summary_csv = accident_out / "predicted_risk_summary.csv"
            risk_script = (
                "predicted_risk_with_speed.py"
                if args.risk_mode == "speed"
                else "predicted_risk_with_pair.py"
            )
            cmd5 = [
                sys.executable,
                str(eval_dir / risk_script),
                "--baseline-csv",
                str(baseline_csv),
                "--base-frame",
                str(t.frame),
                "--base-dir",
                str(t.path),
                "--dt",
                str(args.dt),
                "--use-switch-window",
                "--window-from-switch-sec",
                str(args.window_from_switch_sec),
                "--out-per-run",
                str(risk_per_run_csv),
                "--out-summary",
                str(risk_summary_csv),
            ]
            run_cmd(cmd5, repo_root)

            # 6) summary figures
            run_cmd(
                [
                    sys.executable,
                    str(eval_dir / "plot_results_figures.py"),
                    "--near-miss-collisions",
                    str(per_method_csv),
                    "--risk-summary",
                    str(risk_summary_csv),
                    "--out-collision-pdf",
                    str(accident_out / "fig_collision_summary.pdf"),
                    "--out-risk-pdf",
                    str(accident_out / "fig_risk_summary.pdf"),
                ],
                repo_root,
            )

            index_rows.append(
                {
                    "accident_tag": t.tag,
                    "accident_frame": t.frame,
                    "accident_dir": str(t.path),
                    "status": "ok",
                    "error": "",
                }
            )

            if risk_summary_csv.exists():
                dfr = pd.read_csv(risk_summary_csv)
                dfr.insert(0, "accident_tag", t.tag)
                dfr.insert(1, "accident_frame", t.frame)
                combined_risk_rows.append(dfr)
            if risk_per_run_csv.exists():
                dfr_run = pd.read_csv(risk_per_run_csv)
                dfr_run.insert(0, "accident_tag", t.tag)
                dfr_run.insert(1, "accident_frame", t.frame)
                combined_risk_per_run_rows.append(dfr_run)

            if per_method_csv.exists():
                dfc = pd.read_csv(per_method_csv)
                dfc.insert(0, "accident_tag", t.tag)
                dfc.insert(1, "accident_frame", t.frame)
                combined_coll_rows.append(dfc)
            if per_run_csv.exists():
                dfc_run = pd.read_csv(per_run_csv)
                dfc_run.insert(0, "accident_tag", t.tag)
                dfc_run.insert(1, "accident_frame", t.frame)
                combined_coll_per_run_rows.append(dfc_run)

            print(f"[OK] Completed: {t.tag}")

        except subprocess.CalledProcessError as exc:
            index_rows.append(
                {
                    "accident_tag": t.tag,
                    "accident_frame": t.frame,
                    "accident_dir": str(t.path),
                    "status": "failed",
                    "error": f"command failed: {exc}",
                }
            )
            print(f"[ERROR] Failed: {t.tag}: {exc}")
            if args.fail_fast:
                break

    index_csv = out_root / "analysis_index.csv"
    with index_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["accident_tag", "accident_frame", "accident_dir", "status", "error"],
        )
        w.writeheader()
        w.writerows(index_rows)

    if combined_risk_rows:
        pd.concat(combined_risk_rows, ignore_index=True).to_csv(
            out_root / "predicted_risk_summary_all_accidents.csv", index=False
        )
    if combined_risk_per_run_rows:
        risk_per_run_all = pd.concat(combined_risk_per_run_rows, ignore_index=True)
        risk_per_run_all.to_csv(out_root / "predicted_risk_per_run_all_accidents.csv", index=False)
        risk_global = aggregate_risk_global(risk_per_run_all)
        if not risk_global.empty:
            risk_global.to_csv(out_root / "predicted_risk_summary_global.csv", index=False)

    if combined_coll_rows:
        coll_per_method_all = pd.concat(combined_coll_rows, ignore_index=True)
        coll_per_method_all.to_csv(out_root / "near_miss_and_collisions_per_method_all_accidents.csv", index=False)
        coll_global_like = aggregate_per_method_like_global(coll_per_method_all)
        if not coll_global_like.empty:
            coll_global_like.to_csv(
                out_root / "near_miss_and_collisions_per_method_global.csv",
                index=False,
            )
    if combined_coll_per_run_rows:
        coll_per_run_all = pd.concat(combined_coll_per_run_rows, ignore_index=True)
        coll_per_run_all.to_csv(
            out_root / "near_miss_and_collisions_per_run_all_accidents.csv", index=False
        )
        coll_global = aggregate_collision_global(coll_per_run_all)
        if not coll_global.empty:
            coll_global.to_csv(out_root / "near_miss_and_collisions_summary_global.csv", index=False)

    # 7) Global summary figures from aggregated CSVs (single output across all accidents)
    global_coll_csv = out_root / "near_miss_and_collisions_per_method_global.csv"
    global_risk_csv = out_root / "predicted_risk_summary_global.csv"
    if global_coll_csv.exists() and global_risk_csv.exists():
        run_cmd(
            [
                sys.executable,
                str(eval_dir / "plot_results_figures.py"),
                "--near-miss-collisions",
                str(global_coll_csv),
                "--risk-summary",
                str(global_risk_csv),
                "--out-collision-pdf",
                str(out_root / "fig_collision_summary_global.pdf"),
                "--out-risk-pdf",
                str(out_root / "fig_risk_summary_global.pdf"),
            ],
            repo_root,
        )

    print(f"[DONE] index -> {index_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
