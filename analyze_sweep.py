#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze sweep results:
- per-run trajectory plot (XY)
- RMSE against source CSV
- per-run frame time stats (from stream_timing_*.csv, if present)

Usage:
  python analyze_sweep.py --sweep-root sweep_results_20260130_17304234 --src-csv send_data/exp_300.csv

Outputs:
  <sweep-root>/summary_runs.csv
  <sweep-root>/plots/<RUN_TAG>_traj.png
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Column auto-detection
# -------------------------
def _pick_first(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def detect_frame_col(cols: List[str]) -> Optional[str]:
    # prefer exact matches
    for c in ["frame", "snapshot_frame", "world_frame", "carla_frame"]:
        hit = _pick_first(cols, [c])
        if hit:
            return hit
    # fallback: anything containing "frame"
    for c in cols:
        if "frame" in c.lower():
            return c
    return None


def detect_id_col(cols: List[str]) -> Optional[str]:
    for c in ["carla_actor_id", "actor_id", "id", "object_id", "track_id"]:
        hit = _pick_first(cols, [c])
        if hit:
            return hit
    # fallback: something that ends with "_id"
    for c in cols:
        if c.lower().endswith("_id"):
            return c
    return None


def detect_xyz_cols(cols: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # common exact names
    x = _pick_first(cols, ["x", "location_x", "pos_x", "px"])
    y = _pick_first(cols, ["y", "location_y", "pos_y", "py"])
    z = _pick_first(cols, ["z", "location_z", "pos_z", "pz"])

    # regex fallback: *_x, *_y, *_z
    def find_axis(axis: str) -> Optional[str]:
        pat = re.compile(rf"(^|_){axis}$", re.IGNORECASE)
        for c in cols:
            if pat.search(c):
                return c
        return None

    if x is None:
        x = find_axis("x")
    if y is None:
        y = find_axis("y")
    if z is None:
        z = find_axis("z")

    return x, y, z


def detect_dt_col(cols: List[str]) -> Optional[str]:
    # streamer timing file might have one of these
    for c in [
        "tick_wall_dt_ms",
        "tick_wall_dt",
        "wall_dt_ms",
        "wall_dt",
        "dt_ms",
        "dt",
        "tick_dt_ms",
        "tick_dt",
    ]:
        hit = _pick_first(cols, [c])
        if hit:
            return hit
    # fallback: any column containing "dt" and "ms"
    for c in cols:
        cl = c.lower()
        if "dt" in cl and "ms" in cl:
            return c
    return None


# -------------------------
# Parsing run directory name
# -------------------------
_RUN_RE = re.compile(r"^N(?P<n>\d+)_Ts(?P<ts>[0-9.]+)$", re.IGNORECASE)


@dataclass
class RunInfo:
    tag: str
    n: int
    ts: float
    rundir: Path
    state_csv: Path
    timing_csv: Optional[Path]
    sender_log: Optional[Path]


def discover_runs(sweep_root: Path) -> List[RunInfo]:
    runs: List[RunInfo] = []
    for child in sorted(sweep_root.iterdir()):
        if not child.is_dir():
            continue
        m = _RUN_RE.match(child.name)
        if not m:
            continue

        n = int(m.group("n"))
        ts = float(m.group("ts"))
        # find files
        state = next(child.glob("replay_state_*.csv"), None)
        if state is None:
            # allow alternative naming
            state = next(child.glob("*state*.csv"), None)
        timing = next(child.glob("stream_timing_*.csv"), None)
        if timing is None:
            timing = next(child.glob("*timing*.csv"), None)
        sender = next(child.glob("sender_*.log"), None)

        if state is None:
            continue

        runs.append(
            RunInfo(
                tag=child.name,
                n=n,
                ts=ts,
                rundir=child,
                state_csv=state,
                timing_csv=timing,
                sender_log=sender,
            )
        )
    return runs


# -------------------------
# Core metrics
# -------------------------
def load_source_subset(src_csv: Path, n_max: int) -> pd.DataFrame:
    df = pd.read_csv(src_csv)
    fcol = detect_frame_col(df.columns.tolist())
    icol = detect_id_col(df.columns.tolist())
    xcol, ycol, zcol = detect_xyz_cols(df.columns.tolist())

    missing = [k for k, v in {"frame": fcol, "id": icol, "x": xcol, "y": ycol}.items() if v is None]
    if missing:
        raise ValueError(f"source csv missing required columns (auto-detect failed): {missing}. columns={list(df.columns)}")

    # limit to first n_max ids (sorted) to match --max-actors behavior
    ids = sorted(df[icol].dropna().unique().tolist())
    keep_ids = set(ids[:n_max])
    df = df[df[icol].isin(keep_ids)].copy()

    keep_cols = [fcol, icol, xcol, ycol] + ([zcol] if zcol is not None else [])
    df = df[keep_cols].rename(columns={fcol: "frame", icol: "id", xcol: "x", ycol: "y", **({zcol: "z"} if zcol else {})})
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
    return df.dropna(subset=["frame", "id", "x", "y"])


def load_run_state(state_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(state_csv)
    fcol = detect_frame_col(df.columns.tolist())
    icol = detect_id_col(df.columns.tolist())
    xcol, ycol, zcol = detect_xyz_cols(df.columns.tolist())

    missing = [k for k, v in {"frame": fcol, "id": icol, "x": xcol, "y": ycol}.items() if v is None]
    if missing:
        raise ValueError(f"state csv missing required columns (auto-detect failed): {missing}. file={state_csv} cols={list(df.columns)}")

    keep_cols = [fcol, icol, xcol, ycol] + ([zcol] if zcol is not None else [])
    df = df[keep_cols].rename(columns={fcol: "frame", icol: "id", xcol: "x", ycol: "y", **({zcol: "z"} if zcol else {})})
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
    return df.dropna(subset=["frame", "id", "x", "y"])


def rmse_xy(src: pd.DataFrame, run: pd.DataFrame) -> Tuple[float, int]:
    merged = pd.merge(run, src, on=["frame", "id"], suffixes=("_run", "_src"))
    if merged.empty:
        return float("nan"), 0
    dx = merged["x_run"].to_numpy(dtype=float) - merged["x_src"].to_numpy(dtype=float)
    dy = merged["y_run"].to_numpy(dtype=float) - merged["y_src"].to_numpy(dtype=float)
    e2 = dx * dx + dy * dy
    return float(np.sqrt(np.mean(e2))), int(len(merged))


def rmse_xyz(src: pd.DataFrame, run: pd.DataFrame) -> Tuple[float, int]:
    if "z" not in src.columns or "z" not in run.columns:
        return float("nan"), 0
    merged = pd.merge(run, src, on=["frame", "id"], suffixes=("_run", "_src"))
    if merged.empty:
        return float("nan"), 0
    dx = merged["x_run"].to_numpy(dtype=float) - merged["x_src"].to_numpy(dtype=float)
    dy = merged["y_run"].to_numpy(dtype=float) - merged["y_src"].to_numpy(dtype=float)
    dz = merged["z_run"].to_numpy(dtype=float) - merged["z_src"].to_numpy(dtype=float)
    e2 = dx * dx + dy * dy + dz * dz
    return float(np.sqrt(np.mean(e2))), int(len(merged))


def timing_stats(timing_csv: Optional[Path]) -> Dict[str, float]:
    if timing_csv is None or (not timing_csv.exists()):
        return {"dt_mean_ms": float("nan"), "dt_p50_ms": float("nan"), "dt_p95_ms": float("nan"), "dt_p99_ms": float("nan")}
    df = pd.read_csv(timing_csv)
    dtcol = detect_dt_col(df.columns.tolist())
    if dtcol is None:
        return {"dt_mean_ms": float("nan"), "dt_p50_ms": float("nan"), "dt_p95_ms": float("nan"), "dt_p99_ms": float("nan")}
    dt = pd.to_numeric(df[dtcol], errors="coerce").dropna().to_numpy(dtype=float)
    if dt.size == 0:
        return {"dt_mean_ms": float("nan"), "dt_p50_ms": float("nan"), "dt_p95_ms": float("nan"), "dt_p99_ms": float("nan")}
    return {
        "dt_mean_ms": float(np.mean(dt)),
        "dt_p50_ms": float(np.percentile(dt, 50)),
        "dt_p95_ms": float(np.percentile(dt, 95)),
        "dt_p99_ms": float(np.percentile(dt, 99)),
    }


def plot_xy(run_df: pd.DataFrame, out_png: Path, max_actors: int = 20) -> None:
    import matplotlib.pyplot as plt

    ids = sorted(run_df["id"].dropna().unique().tolist())[:max_actors]
    plt.figure()
    for aid in ids:
        sub = run_df[run_df["id"] == aid].sort_values("frame")
        plt.plot(sub["x"].to_numpy(), sub["y"].to_numpy(), linewidth=1)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(out_png.stem)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


# -------------------------
# Main
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", required=True, help="sweep_results_... directory")
    ap.add_argument("--src-csv", required=True, help="source (sender) CSV, e.g. send_data/exp_300.csv")
    ap.add_argument("--plot", action="store_true", help="write trajectory plots")
    ap.add_argument("--plot-max-actors", type=int, default=20, help="max actors per plot")
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root).resolve()
    src_csv = Path(args.src_csv).resolve()

    runs = discover_runs(sweep_root)
    if not runs:
        raise SystemExit(f"no runs found under: {sweep_root}")

    rows = []
    plots_dir = sweep_root / "plots"

    for r in runs:
        run_df = load_run_state(r.state_csv)
        src_df = load_source_subset(src_csv, r.n)

        rmse2d, nmatch2d = rmse_xy(src_df, run_df)
        rmse3d, nmatch3d = rmse_xyz(src_df, run_df)

        tstats = timing_stats(r.timing_csv)

        if args.plot:
            out_png = plots_dir / f"{r.tag}_traj.png"
            plot_xy(run_df, out_png, max_actors=args.plot_max_actors)

        rows.append(
            {
                "tag": r.tag,
                "N": r.n,
                "Ts": r.ts,
                "state_csv": str(r.state_csv),
                "timing_csv": str(r.timing_csv) if r.timing_csv else "",
                "rmse_xy": rmse2d,
                "rmse_xy_matched_samples": nmatch2d,
                "rmse_xyz": rmse3d,
                "rmse_xyz_matched_samples": nmatch3d,
                **tstats,
            }
        )

    out_csv = sweep_root / "summary_runs.csv"
    pd.DataFrame(rows).sort_values(["N", "Ts"]).to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")
    if args.plot:
        print(f"Wrote plots to: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
