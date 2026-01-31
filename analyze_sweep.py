#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze CARLA UDP replay sweep results.

Per run (e.g., N10_Ts0.10):
  - Plot trajectories to PDF (single blue color, no color-coding)
  - Compute RMSE vs reference (exp_300.csv)
  - Summarize per-frame timing stats (from timing CSV if present)

Expected directory layout:
  sweep_results_XXXX/
    update_timings_all.csv              (optional, global)
    eval_all.csv                        (optional, global)
    N10_Ts0.10/
      replay_state_N10_Ts0.10.csv        (from vehicle_state_stream.py)
      stream_timing_N10_Ts0.10.csv       (from vehicle_state_stream.py timing)
      ...

Usage:
  python analyze_sweep.py --outdir sweep_results_20260130_17304234 --ref send_data/exp_300.csv --paper
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# -----------------------------
# Column heuristics
# -----------------------------
FRAME_CANDS = ["frame", "Frame", "pf", "payload_frame", "payloadFrame", "step"]
ID_CANDS = ["carla_actor_id", "actor_id", "id", "object_id", "tracking_id", "track_id"]
X_CANDS = ["location_x", "x", "pos_x", "px", "X"]
Y_CANDS = ["location_y", "y", "pos_y", "py", "Y"]
TYPE_CANDS = ["type", "actor_type", "kind", "category"]

# timing candidates: pick first existing numeric column
TIMING_CANDS = [
    "frame_ms",
    "elapsed_ms",
    "process_ms",
    "processing_ms",
    "update_ms",
    "tick_wall_dt_ms",
    "tick_wall_dt",
    "dt_ms",
    "dt",
]


def _pick_col(cols: Iterable[str], candidates: List[str]) -> Optional[str]:
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols_l[cand.lower()]
    return None


def _normalize_positions(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Return df with columns: actor_id(int), frame(int), x(float), y(float), type(optional str)."""
    fcol = _pick_col(df.columns, FRAME_CANDS)
    icol = _pick_col(df.columns, ID_CANDS)
    xcol = _pick_col(df.columns, X_CANDS)
    ycol = _pick_col(df.columns, Y_CANDS)
    tcol = _pick_col(df.columns, TYPE_CANDS)

    missing = [k for k, v in [("frame", fcol), ("id", icol), ("x", xcol), ("y", ycol)] if v is None]
    if missing:
        raise ValueError(
            f"{source_name}: required columns not found: {missing}. "
            f"available={list(df.columns)}"
        )

    out = pd.DataFrame()
    out["frame"] = pd.to_numeric(df[fcol], errors="coerce")
    out["actor_id"] = pd.to_numeric(df[icol], errors="coerce")
    out["x"] = pd.to_numeric(df[xcol], errors="coerce")
    out["y"] = pd.to_numeric(df[ycol], errors="coerce")
    if tcol is not None:
        out["type"] = df[tcol].astype(str)
    else:
        out["type"] = ""

    out = out.dropna(subset=["frame", "actor_id", "x", "y"]).copy()
    # frame is typically integer payload frame. keep int for safe merge
    out["frame"] = out["frame"].round().astype(int)
    out["actor_id"] = out["actor_id"].round().astype(int)
    return out


def load_positions(csv_path: Path) -> pd.DataFrame:
    # robust encoding
    for enc in ["utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            return _normalize_positions(df, str(csv_path))
        except UnicodeError:
            continue
    # last attempt: default
    df = pd.read_csv(csv_path)
    return _normalize_positions(df, str(csv_path))


def compute_rmse(ref: pd.DataFrame, run: pd.DataFrame) -> Tuple[float, float, float, int]:
    """
    RMSE_x, RMSE_y, RMSE_xy, n_samples
    RMSE_xy uses sqrt(mean(dx^2 + dy^2)).
    """
    merged = pd.merge(
        ref[["actor_id", "frame", "x", "y"]],
        run[["actor_id", "frame", "x", "y"]],
        on=["actor_id", "frame"],
        suffixes=("_ref", "_run"),
        how="inner",
    )
    if merged.empty:
        return float("nan"), float("nan"), float("nan"), 0
    dx = merged["x_run"] - merged["x_ref"]
    dy = merged["y_run"] - merged["y_ref"]
    rmse_x = math.sqrt(float((dx * dx).mean()))
    rmse_y = math.sqrt(float((dy * dy).mean()))
    rmse_xy = math.sqrt(float(((dx * dx + dy * dy)).mean()))
    return rmse_x, rmse_y, rmse_xy, int(len(merged))


def compute_rmse_per_actor(ref: pd.DataFrame, run: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        ref[["actor_id", "frame", "x", "y"]],
        run[["actor_id", "frame", "x", "y"]],
        on=["actor_id", "frame"],
        suffixes=("_ref", "_run"),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=["actor_id", "n", "rmse_x", "rmse_y", "rmse_xy"])
    merged["dx2"] = (merged["x_run"] - merged["x_ref"]) ** 2
    merged["dy2"] = (merged["y_run"] - merged["y_ref"]) ** 2
    g = merged.groupby("actor_id", as_index=False).agg(
        n=("frame", "count"),
        mse_x=("dx2", "mean"),
        mse_y=("dy2", "mean"),
        mse_xy=(["dx2", "dy2"], "mean"),  # placeholder, overwritten below
    )
    # pandas gives multiindex for mse_xy; easier manual
    per = []
    for aid, grp in merged.groupby("actor_id"):
        mse_x = float(grp["dx2"].mean())
        mse_y = float(grp["dy2"].mean())
        mse_xy = float((grp["dx2"] + grp["dy2"]).mean())
        per.append(
            {
                "actor_id": int(aid),
                "n": int(len(grp)),
                "rmse_x": math.sqrt(mse_x),
                "rmse_y": math.sqrt(mse_y),
                "rmse_xy": math.sqrt(mse_xy),
            }
        )
    return pd.DataFrame(per).sort_values(["actor_id"]).reset_index(drop=True)


def load_timing_ms(timing_csv: Path) -> Optional[pd.Series]:
    try:
        df = pd.read_csv(timing_csv, encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(timing_csv)
        except Exception:
            return None

    col = _pick_col(df.columns, TIMING_CANDS)
    if col is None:
        # fallback: first numeric column
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            return None
        col = num_cols[0]

    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None

    # if unit seems seconds (very small), convert to ms
    if s.median() < 0.5:
        s = s * 1000.0
    return s


def timing_stats(s: Optional[pd.Series]) -> Dict[str, float]:
    if s is None or s.empty:
        return {
            "frame_time_ms_mean": float("nan"),
            "frame_time_ms_p50": float("nan"),
            "frame_time_ms_p90": float("nan"),
            "frame_time_ms_p99": float("nan"),
            "frame_time_ms_max": float("nan"),
            "frame_time_n": 0.0,
        }
    return {
        "frame_time_ms_mean": float(s.mean()),
        "frame_time_ms_p50": float(s.quantile(0.50)),
        "frame_time_ms_p90": float(s.quantile(0.90)),
        "frame_time_ms_p99": float(s.quantile(0.99)),
        "frame_time_ms_max": float(s.max()),
        "frame_time_n": float(len(s)),
    }


def setup_fonts(paper: bool) -> None:
    if paper:
        mpl.rcParams["font.family"] = [
            "BIZ UDP Gothic",
            "BIZ UDPゴシック",
            "IPAexGothic",
            "MS Gothic",
            "sans-serif",
        ]


def plot_run_pdf(
    ref: pd.DataFrame,
    run: pd.DataFrame,
    out_pdf: Path,
    title: str,
    paper: bool,
) -> None:
    setup_fonts(paper)

    # single blue, no color-coding
    blue = "#1f77b4"  # tab:blue
    lw = 1.8 if paper else 1.5
    alpha = 0.85 if paper else 1.0

    fig, ax = plt.subplots(figsize=(10, 8))

    # plot reference as dashed, same color
    for aid, grp in ref.groupby("actor_id"):
        grp = grp.sort_values("frame")
        ax.plot(grp["x"], grp["y"], color=blue, linewidth=lw, alpha=0.35, linestyle="--")

    # plot run as solid
    for aid, grp in run.groupby("actor_id"):
        grp = grp.sort_values("frame")
        ax.plot(grp["x"], grp["y"], color=blue, linewidth=lw, alpha=alpha, linestyle="-")

    label_fs = 22 if paper else None
    tick_fs = 20 if paper else None

    ax.set_xlabel("X [m]", fontsize=label_fs)
    ax.set_ylabel("Y [m]", fontsize=label_fs)
    if tick_fs:
        ax.tick_params(labelsize=tick_fs)
        ax.tick_params(direction="in")

    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.35)

    if not paper:
        ax.set_title(title)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=200)
    plt.close(fig)


def find_run_dirs(outdir: Path) -> List[Path]:
    # match N10_Ts0.10 etc
    runs = []
    for p in outdir.iterdir():
        if p.is_dir() and re.match(r"^N\d+_Ts\d+(\.\d+)?$", p.name):
            runs.append(p)
    return sorted(runs, key=lambda x: x.name)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True, help="sweep_results_YYYY... directory")
    ap.add_argument("--ref", type=Path, required=True, help="reference CSV, e.g., send_data/exp_300.csv")
    ap.add_argument("--paper", action="store_true", help="paper styling and font setup")
    ap.add_argument("--per-actor", action="store_true", help="also write per-actor RMSE CSV per run")
    ap.add_argument("--no-pdf", action="store_true", help="skip PDF trajectory outputs")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    outdir: Path = args.outdir
    ref_csv: Path = args.ref

    if not outdir.exists():
        raise SystemExit(f"outdir not found: {outdir}")
    if not ref_csv.exists():
        raise SystemExit(f"ref csv not found: {ref_csv}")

    ref = load_positions(ref_csv)

    run_dirs = find_run_dirs(outdir)
    if not run_dirs:
        raise SystemExit(f"no run dirs found under: {outdir}")

    rows = []
    for rd in run_dirs:
        tag = rd.name

        # locate run trajectory csv
        cand_traj = sorted(rd.glob("replay_state_*.csv"))
        if not cand_traj:
            # fallback: any csv that looks like state stream output
            cand_traj = sorted(rd.glob("*.csv"))
        traj_csv = None
        for c in cand_traj:
            if "replay_state" in c.name:
                traj_csv = c
                break
        if traj_csv is None and cand_traj:
            traj_csv = cand_traj[0]
        if traj_csv is None:
            continue

        run = load_positions(traj_csv)

        # RMSE
        rmse_x, rmse_y, rmse_xy, n = compute_rmse(ref, run)

        # timing (prefer per-run stream timing)
        timing_csv = None
        for c in sorted(rd.glob("stream_timing_*.csv")):
            timing_csv = c
            break
        timing_s = load_timing_ms(timing_csv) if timing_csv else None
        tstat = timing_stats(timing_s)

        # PDF
        pdf_path = rd / f"traj_{tag}.pdf"
        if not args.no_pdf:
            title = f"{tag}  RMSE_xy={rmse_xy:.3f} m  (n={n})"
            plot_run_pdf(ref, run, pdf_path, title=title, paper=args.paper)

        # per-actor RMSE
        if args.per_actor:
            per = compute_rmse_per_actor(ref, run)
            per.to_csv(rd / f"rmse_per_actor_{tag}.csv", index=False)

        rows.append(
            {
                "tag": tag,
                "traj_csv": str(traj_csv),
                "rmse_x_m": rmse_x,
                "rmse_y_m": rmse_y,
                "rmse_xy_m": rmse_xy,
                "rmse_n": n,
                "timing_csv": str(timing_csv) if timing_csv else "",
                **tstat,
                "pdf": str(pdf_path) if (not args.no_pdf) else "",
            }
        )

    summary = pd.DataFrame(rows).sort_values("tag")
    summary_path = outdir / "summary_runs.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote: {summary_path}")
    if not args.no_pdf:
        print("Per-run PDFs: traj_<TAG>.pdf under each run directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
