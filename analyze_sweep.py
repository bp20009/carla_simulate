#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze CARLA sweep results.

What it generates (PDF only):
- Run trajectories per run (no color split, blue-ish)
- Reference trajectories for N=10..100 step 10 (separate figures, not overlaid)
- RMSE per actor_id and per run (shape-based, NOT frame-based)
- Per-run "frame time" time-series plot (elapsed from 0) + summary stats

RMSE matching policy:
- Do NOT align by frame or wall time.
- Align by trajectory arc-length (normalized 0..1) and resample to fixed points.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ---------- plotting defaults ----------
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def pick_font(prefer_biz_udp: bool = True) -> str:
    """
    Pick an installed font name to avoid findfont spam.
    Priority: BIZ UDP Gothic -> Arial -> DejaVu Sans
    """
    installed = {f.name for f in font_manager.fontManager.ttflist}
    if prefer_biz_udp:
        for cand in ["BIZ UDP Gothic", "BIZ UDPゴシック"]:
            if cand in installed:
                return cand
    for cand in ["Arial", "DejaVu Sans"]:
        if cand in installed:
            return cand
    # last resort: let matplotlib decide
    return "sans-serif"


def configure_paper_fonts() -> None:
    font = pick_font(prefer_biz_udp=True)
    mpl.rcParams["font.family"] = font
    # Keep it explicit for reproducibility
    print(f"[INFO] Using font: {font}")


# ---------- IO helpers ----------
def read_csv_flexible(path: Path) -> pd.DataFrame:
    """
    Read CSV with robust encoding fallback and NUL stripping.
    """
    # Try encodings in order
    encodings = ["utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp932"]
    last_err: Optional[Exception] = None
    raw: Optional[str] = None

    for enc in encodings:
        try:
            raw = path.read_text(encoding=enc, errors="strict")
            break
        except Exception as e:
            last_err = e

    if raw is None:
        raise RuntimeError(f"Failed to read {path} with encodings {encodings}: {last_err}")

    raw = raw.replace("\x00", "")
    # Use pandas via StringIO
    from io import StringIO
    return pd.read_csv(StringIO(raw))


def ensure_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}. columns={list(df.columns)}")


# ---------- trajectory + RMSE ----------
@dataclass
class Traj:
    x: np.ndarray
    y: np.ndarray

    def valid(self) -> bool:
        return len(self.x) >= 2 and len(self.y) == len(self.x)


def traj_from_df(df: pd.DataFrame, xcol: str, ycol: str) -> Traj:
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    # drop NaN pairs
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    return Traj(x=x, y=y)


def resample_by_arclength(traj: Traj, n_points: int = 200) -> Traj:
    """
    Resample a polyline by normalized arc-length in [0,1].
    """
    x, y = traj.x, traj.y
    if len(x) < 2:
        return traj

    dx = np.diff(x)
    dy = np.diff(y)
    seg = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])

    if total <= 1e-12:
        # no movement: keep as constant
        xx = np.full(n_points, x[0], dtype=float)
        yy = np.full(n_points, y[0], dtype=float)
        return Traj(xx, yy)

    u = s / total
    u_new = np.linspace(0.0, 1.0, n_points)

    x_new = np.interp(u_new, u, x)
    y_new = np.interp(u_new, u, y)
    return Traj(x_new, y_new)


def rmse_xy(a: Traj, b: Traj) -> Tuple[float, float, float]:
    """
    Returns (rmse_x, rmse_y, rmse_xy) where rmse_xy is RMSE over Euclidean distance.
    """
    if len(a.x) != len(b.x):
        raise ValueError("rmse_xy: trajectories must be same length after resampling")

    dx = a.x - b.x
    dy = a.y - b.y
    rmse_x = float(np.sqrt(np.mean(dx * dx)))
    rmse_y = float(np.sqrt(np.mean(dy * dy)))
    rmse_e = float(np.sqrt(np.mean(dx * dx + dy * dy)))
    return rmse_x, rmse_y, rmse_e


# ---------- time-series (frame time) ----------
def discover_frame_time_series(run_dir: Path) -> Optional[pd.DataFrame]:
    """
    Try to discover a per-frame time series to plot.
    Priority:
      1) stream_timing_*.csv (vehicle_state_stream --timing-output)
      2) receiver_stderr.log within run_dir (if you have per-run logs)
    Output DataFrame columns: elapsed_s, frame_time_ms
    """
    # 1) stream_timing_*.csv
    timing = None
    cand = list(run_dir.glob("stream_timing_*.csv"))
    if cand:
        # choose the largest one (most samples)
        cand.sort(key=lambda p: p.stat().st_size, reverse=True)
        try:
            timing = read_csv_flexible(cand[0])
        except Exception:
            timing = None

    if timing is not None and not timing.empty:
        # Heuristics: pick a time step column
        cols = list(timing.columns)

        # elapsed: prefer monotonic or elapsed columns if exist
        time_candidates = [
            "elapsed_s", "elapsed", "frame_elapsed", "wall_clock", "wall_time",
            "monotonic", "t", "time", "timestamp"
        ]
        tcol = next((c for c in time_candidates if c in cols), None)

        # frame-time candidates in seconds or ms
        ms_candidates = [c for c in cols if re.search(r"(ms|msec)", c, re.IGNORECASE)]
        dt_candidates = [c for c in cols if re.search(r"(dt|delta)", c, re.IGNORECASE)]

        ycol = None
        scale = 1.0  # to ms

        if ms_candidates:
            # pick something that looks like processing or tick
            preferred = [c for c in ms_candidates if re.search(r"(tick|loop|frame|proc|step)", c, re.IGNORECASE)]
            ycol = preferred[0] if preferred else ms_candidates[0]
            scale = 1.0
        elif dt_candidates:
            preferred = [c for c in dt_candidates if re.search(r"(tick|wall|loop|frame)", c, re.IGNORECASE)]
            ycol = preferred[0] if preferred else dt_candidates[0]
            # guess unit: if median < 2.0, assume seconds -> ms
            med = float(np.nanmedian(pd.to_numeric(timing[ycol], errors="coerce")))
            scale = 1000.0 if med < 2.0 else 1.0

        if ycol is not None:
            y = pd.to_numeric(timing[ycol], errors="coerce") * scale
            if tcol is not None:
                t = pd.to_numeric(timing[tcol], errors="coerce")
                t = t - float(np.nanmin(t))
            else:
                # fallback: build elapsed from cumulative dt if available
                if "tick_wall_dt" in cols:
                    dt = pd.to_numeric(timing["tick_wall_dt"], errors="coerce").fillna(0.0)
                    t = dt.cumsum()
                else:
                    # assume 1 step index
                    t = pd.Series(np.arange(len(y), dtype=float))
            out = pd.DataFrame({"elapsed_s": t, "frame_time_ms": y})
            out = out.replace([np.inf, -np.inf], np.nan).dropna()
            if len(out) >= 5:
                return out

    # 2) per-run receiver_stderr.log style parsing (optional)
    log = run_dir / "receiver_stderr.log"
    if log.exists():
        txt = log.read_text(encoding="utf-8", errors="ignore")
        # Extract "Frame processed in XX ms"
        vals = [float(m.group(1)) for m in re.finditer(r"Frame processed in\s+([0-9.]+)\s*ms", txt)]
        if len(vals) >= 5:
            y = np.array(vals, dtype=float)
            # elapsed = index * median_dt (unknown) -> just index-based seconds
            x = np.arange(len(y), dtype=float)
            # normalize x to start 0, treat as seconds-ish
            out = pd.DataFrame({"elapsed_s": x, "frame_time_ms": y})
            return out

    return None


# ---------- plotting ----------
def save_traj_pdf(path: Path, trajs: Dict[int, Traj], title: str, color: str, paper: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    for actor_id in sorted(trajs.keys()):
        tr = trajs[actor_id]
        if not tr.valid():
            continue
        ax.plot(tr.x, tr.y, linewidth=1.6 if paper else 1.2, alpha=0.85)

    ax.set_xlabel("X [m]", fontsize=22 if paper else None)
    ax.set_ylabel("Y [m]", fontsize=22 if paper else None)
    ax.tick_params(labelsize=20 if paper else None)
    ax.tick_params(direction="in")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.35)
    if not paper:
        ax.set_title(title)

    # Set a single artist color after plotting (color cycle override)
    for line in ax.lines:
        line.set_color(color)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_frame_time_pdf(path: Path, df: pd.DataFrame, title: str, paper: bool) -> Dict[str, float]:
    """
    df: columns elapsed_s, frame_time_ms
    returns summary stats
    """
    x = df["elapsed_s"].to_numpy(dtype=float)
    y = df["frame_time_ms"].to_numpy(dtype=float)

    # Ensure x starts at 0
    x = x - float(np.min(x))

    stats = {
        "n": float(len(y)),
        "mean_ms": float(np.mean(y)),
        "median_ms": float(np.median(y)),
        "p90_ms": float(np.quantile(y, 0.90)),
        "p99_ms": float(np.quantile(y, 0.99)),
        "max_ms": float(np.max(y)),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, linewidth=1.4, alpha=0.9)
    ax.set_xlabel("Elapsed time [s]", fontsize=20 if paper else None)
    ax.set_ylabel("Frame time [ms]", fontsize=20 if paper else None)
    ax.tick_params(labelsize=18 if paper else None)
    ax.tick_params(direction="in")
    ax.grid(True, linestyle="--", alpha=0.35)
    if not paper:
        ax.set_title(title)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return stats


# ---------- sweep discovery ----------
RUN_DIR_RE = re.compile(r"^N(?P<n>\d+)_Ts(?P<ts>[0-9.]+)$")


def parse_run_tag(name: str) -> Optional[Tuple[int, float]]:
    m = RUN_DIR_RE.match(name)
    if not m:
        return None
    return int(m.group("n")), float(m.group("ts"))


def find_run_dirs(outdir: Path) -> List[Path]:
    runs = []
    for p in outdir.iterdir():
        if p.is_dir() and parse_run_tag(p.name) is not None:
            runs.append(p)
    runs.sort(key=lambda d: (parse_run_tag(d.name)[0], parse_run_tag(d.name)[1]))  # type: ignore
    return runs


def load_run_trajectories(run_dir: Path) -> Dict[int, Traj]:
    """
    Load replay_state_*.csv in run_dir.
    Expect columns similar to vehicle_state_stream output:
      - actor identifier: carla_actor_id or id
      - location: location_x, location_y
    """
    cands = list(run_dir.glob("replay_state_*.csv"))
    if not cands:
        raise FileNotFoundError(f"No replay_state_*.csv in {run_dir}")
    cands.sort(key=lambda p: p.stat().st_size, reverse=True)
    df = read_csv_flexible(cands[0])

    # Heuristic column mapping
    id_col = "carla_actor_id" if "carla_actor_id" in df.columns else "id"
    xcol = "location_x" if "location_x" in df.columns else ("x" if "x" in df.columns else None)
    ycol = "location_y" if "location_y" in df.columns else ("y" if "y" in df.columns else None)
    if xcol is None or ycol is None:
        raise ValueError(f"{cands[0]}: cannot find x/y columns. columns={list(df.columns)}")

    # Optional filter: vehicles only if 'type' exists
    if "type" in df.columns:
        df = df[df["type"].astype(str).str.startswith("vehicle")]

    df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[id_col, xcol, ycol])
    df = df.sort_values(by=[id_col])

    trajs: Dict[int, Traj] = {}
    for actor_id, g in df.groupby(id_col):
        if pd.isna(actor_id):
            continue
        aid = int(actor_id)
        tr = traj_from_df(g, xcol, ycol)
        if tr.valid():
            trajs[aid] = tr
    return trajs


def load_ref_trajectories(ref_csv: Path) -> Dict[int, Traj]:
    """
    Load reference exp_300.csv columns:
      frame,id,type,x,y,z
    """
    df = read_csv_flexible(ref_csv)
    ensure_columns(df, ["id", "x", "y"], "ref")
    if "type" in df.columns:
        df = df[df["type"].astype(str).str.startswith("vehicle")]

    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["id", "x", "y"])
    df = df.sort_values(by=["id", "frame"] if "frame" in df.columns else ["id"])

    trajs: Dict[int, Traj] = {}
    for actor_id, g in df.groupby("id"):
        if pd.isna(actor_id):
            continue
        aid = int(actor_id)
        tr = traj_from_df(g, "x", "y")
        if tr.valid():
            trajs[aid] = tr
    return trajs


def compute_rmse_tables(
    ref_trajs: Dict[int, Traj],
    run_trajs: Dict[int, Traj],
    n_points: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    rmses = []

    common_ids = sorted(set(ref_trajs.keys()) & set(run_trajs.keys()))
    for aid in common_ids:
        a = resample_by_arclength(ref_trajs[aid], n_points=n_points)
        b = resample_by_arclength(run_trajs[aid], n_points=n_points)
        rx, ry, re = rmse_xy(a, b)
        rows.append(
            {
                "actor_id": aid,
                "rmse_x": rx,
                "rmse_y": ry,
                "rmse_xy": re,
                "ref_points": int(len(ref_trajs[aid].x)),
                "run_points": int(len(run_trajs[aid].x)),
            }
        )
        rmses.append(re)

    per_actor = pd.DataFrame(rows)
    summary = {
        "n_common": float(len(common_ids)),
        "rmse_xy_mean": float(np.mean(rmses)) if rmses else float("nan"),
        "rmse_xy_median": float(np.median(rmses)) if rmses else float("nan"),
        "rmse_xy_p90": float(np.quantile(rmses, 0.90)) if rmses else float("nan"),
        "rmse_xy_max": float(np.max(rmses)) if rmses else float("nan"),
    }
    return per_actor, summary


# ---------- main ----------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True, help="sweep_results_... directory")
    ap.add_argument("--ref", type=Path, required=True, help="Reference CSV (exp_300.csv)")
    ap.add_argument("--paper", action="store_true", help="Paper style + font selection")
    ap.add_argument("--color", default="#2a7fb8", help="Single line color (e.g. blue-ish)")
    ap.add_argument("--resample-points", type=int, default=200, help="Points for arc-length resampling")
    ap.add_argument("--ref-n-min", type=int, default=10)
    ap.add_argument("--ref-n-max", type=int, default=100)
    ap.add_argument("--ref-n-step", type=int, default=10)
    args = ap.parse_args()

    outdir: Path = args.outdir
    ref_csv: Path = args.ref

    if args.paper:
        configure_paper_fonts()

    if not outdir.exists():
        raise FileNotFoundError(outdir)
    if not ref_csv.exists():
        raise FileNotFoundError(ref_csv)

    # Load reference once
    ref_trajs_all = load_ref_trajectories(ref_csv)

    # Write ref plots N=10..100
    ref_plot_dir = outdir / "plots_ref"
    for n in range(args.ref_n_min, args.ref_n_max + 1, args.ref_n_step):
        ids = [aid for aid in sorted(ref_trajs_all.keys()) if aid <= n]
        trajs = {aid: ref_trajs_all[aid] for aid in ids}
        save_traj_pdf(
            ref_plot_dir / f"ref_N{n:03d}.pdf",
            trajs,
            title=f"Reference trajectories (N={n})",
            color=args.color,
            paper=args.paper,
        )

    # Run analysis per run
    runs = find_run_dirs(outdir)
    if not runs:
        print("[WARN] No run directories found like N10_Ts0.10 under outdir.")
        return 0

    per_actor_rows = []
    summary_rows = []

    run_plot_dir = outdir / "plots_run"
    time_plot_dir = outdir / "plots_time"

    for run_dir in runs:
        tag = run_dir.name
        parsed = parse_run_tag(tag)
        assert parsed is not None
        n, ts = parsed

        # Load run traj
        run_trajs = load_run_trajectories(run_dir)

        # Plot run trajectories
        save_traj_pdf(
            run_plot_dir / f"{tag}_traj.pdf",
            run_trajs,
            title=f"Run trajectories ({tag})",
            color=args.color,
            paper=args.paper,
        )

        # RMSE per actor (shape-based, id match)
        per_actor, rmse_sum = compute_rmse_tables(
            ref_trajs=ref_trajs_all,
            run_trajs=run_trajs,
            n_points=args.resample_points,
        )
        if not per_actor.empty:
            per_actor.insert(0, "run_tag", tag)
            per_actor.insert(1, "N", n)
            per_actor.insert(2, "Ts", ts)
            per_actor_rows.append(per_actor)

        # Frame time series
        ts_df = discover_frame_time_series(run_dir)
        time_stats = {}
        if ts_df is not None:
            time_stats = save_frame_time_pdf(
                time_plot_dir / f"{tag}_frame_time.pdf",
                ts_df,
                title=f"Frame time ({tag})",
                paper=args.paper,
            )

        # Summary row
        row = {
            "run_tag": tag,
            "N": n,
            "Ts": ts,
            **rmse_sum,
            **time_stats,
        }
        summary_rows.append(row)

        print(f"[OK] {tag}: n_common={rmse_sum.get('n_common')} rmse_mean={rmse_sum.get('rmse_xy_mean')}")

    # Write tables
    summary_df = pd.DataFrame(summary_rows).sort_values(by=["N", "Ts"])
    summary_path = outdir / "summary_runs.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Wrote: {summary_path}")

    if per_actor_rows:
        per_actor_df = pd.concat(per_actor_rows, ignore_index=True)
        per_actor_path = outdir / "rmse_per_actor.csv"
        per_actor_df.to_csv(per_actor_path, index=False, encoding="utf-8-sig")
        print(f"[OK] Wrote: {per_actor_path}")

    print(f"[OK] Runs: {len(runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
