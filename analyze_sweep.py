#!/usr/bin/env python3
"""
Analyze sweep results.

Expected structure:
  <outdir>/
    receiver_*.log ...
    N10_Ts0.10/
      replay_state.csv
      stream_timing.csv
      sender.log
      ...
    N20_Ts0.10/
      ...

Reference CSV (example):
  frame  id  type  x  y  z

This script generates a NEW analysis directory under --outdir and writes:
  - per-run trajectory PDFs (single color)
  - reference trajectory PDFs per N (separate, not overlaid)
  - per-run timing plots and summary stats
  - per-run RMSE (overall + per-actor), matching by id (NOT by frame)
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

# Embed TrueType fonts in PDF (important for papers)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# ----------------------------
# CSV helpers (robust encoding)
# ----------------------------
def _strip_null_bytes(lines: Iterable[str]) -> Iterator[str]:
    for line in lines:
        yield line.replace("\x00", "")


def read_csv_flexible(path: Path) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp932")
    last_exc: Optional[Exception] = None
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as fh:
                reader = csv.reader(_strip_null_bytes(fh))
                rows = list(reader)
            if not rows:
                return pd.DataFrame()
            header = rows[0]
            data = rows[1:]
            return pd.DataFrame(data, columns=header)
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to read CSV with fallback encodings: {path}") from last_exc


def to_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ----------------------------
# Column normalization
# ----------------------------
def normalize_ref(df: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: frame,id,type,x,y,z (as user wrote)
    # Keep only needed and coerce numeric
    required_any = {"frame", "id", "type"}
    if not required_any.issubset(set(df.columns)):
        raise ValueError(f"ref missing required columns {required_any}. got={list(df.columns)}")

    # x,y can be named x,y or location_x,location_y
    if "x" not in df.columns and "location_x" in df.columns:
        df = df.rename(columns={"location_x": "x"})
    if "y" not in df.columns and "location_y" in df.columns:
        df = df.rename(columns={"location_y": "y"})
    if "z" not in df.columns and "location_z" in df.columns:
        df = df.rename(columns={"location_z": "z"})

    df = to_numeric_df(df, ["frame", "id", "x", "y", "z"])
    df = df.dropna(subset=["id", "x", "y"])
    df["id"] = df["id"].astype(int)

    # Filter vehicles by default (type column: 'vehicle' etc.)
    df["type"] = df["type"].astype(str)
    return df


def normalize_run(df: pd.DataFrame) -> pd.DataFrame:
    # replay_state.csv is produced by vehicle_state_stream.py
    # Try to map common column names.
    colmap = {}

    # id
    if "id" in df.columns:
        colmap["id"] = "id"
    elif "actor_id" in df.columns:
        colmap["actor_id"] = "id"
    elif "carla_actor_id" in df.columns:
        colmap["carla_actor_id"] = "id"

    # type
    if "type" in df.columns:
        colmap["type"] = "type"
    elif "actor_type" in df.columns:
        colmap["actor_type"] = "type"

    # frame
    if "frame" in df.columns:
        colmap["frame"] = "frame"
    elif "timestamp" in df.columns:
        colmap["timestamp"] = "frame"  # fallback meaning

    # position
    if "location_x" in df.columns:
        colmap["location_x"] = "x"
    elif "x" in df.columns:
        colmap["x"] = "x"

    if "location_y" in df.columns:
        colmap["location_y"] = "y"
    elif "y" in df.columns:
        colmap["y"] = "y"

    if "location_z" in df.columns:
        colmap["location_z"] = "z"
    elif "z" in df.columns:
        colmap["z"] = "z"

    df = df.rename(columns=colmap)

    required = {"id", "x", "y"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"run CSV missing required columns {required}. got={list(df.columns)}")

    # frame is optional (we sort by it if present)
    num_cols = ["id", "x", "y"]
    if "frame" in df.columns:
        num_cols.append("frame")
    if "z" in df.columns:
        num_cols.append("z")

    df = to_numeric_df(df, num_cols)
    df = df.dropna(subset=["id", "x", "y"])
    df["id"] = df["id"].astype(int)

    if "type" in df.columns:
        df["type"] = df["type"].astype(str)
    else:
        df["type"] = "unknown"

    if "frame" in df.columns:
        df = df.sort_values(["id", "frame"])
    else:
        df = df.sort_values(["id"])
    return df


# ----------------------------
# Trajectory extraction
# ----------------------------
def trajectories_by_id(df: pd.DataFrame, kind: str = "vehicle") -> Dict[int, np.ndarray]:
    """
    Return dict: id -> array[[x,y], ...] ordered.
    kind: filter by df['type'] prefix match (vehicle/walker) if present.
    """
    if "type" in df.columns and kind:
        sub = df[df["type"].astype(str).str.startswith(f"{kind}")]
    else:
        sub = df

    trajs: Dict[int, np.ndarray] = {}
    for aid, g in sub.groupby("id", sort=True):
        # Keep only finite points
        pts = g[["x", "y"]].to_numpy(dtype=float)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) >= 2:
            trajs[int(aid)] = pts
    return trajs


# ----------------------------
# RMSE (id match, time-normalized alignment)
# ----------------------------
def rmse_time_normalized(ref_xy: np.ndarray, run_xy: np.ndarray, samples: int = 200) -> Tuple[float, float, float]:
    """
    Align by normalized progress t in [0,1] with interpolation, then compute RMSE.
    Returns (rmse_2d, rmse_x, rmse_y).
    """
    m = len(ref_xy)
    n = len(run_xy)
    if m < 2 or n < 2:
        return (float("nan"), float("nan"), float("nan"))

    t_ref = np.linspace(0.0, 1.0, m)
    t_run = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 1.0, samples)

    ref_x = np.interp(t, t_ref, ref_xy[:, 0])
    ref_y = np.interp(t, t_ref, ref_xy[:, 1])
    run_x = np.interp(t, t_run, run_xy[:, 0])
    run_y = np.interp(t, t_run, run_xy[:, 1])

    dx = run_x - ref_x
    dy = run_y - ref_y

    rmse_x = float(np.sqrt(np.mean(dx * dx)))
    rmse_y = float(np.sqrt(np.mean(dy * dy)))
    rmse_2d = float(np.sqrt(np.mean(dx * dx + dy * dy)))
    return rmse_2d, rmse_x, rmse_y


def compute_rmse_per_actor(ref_trajs: Dict[int, np.ndarray], run_trajs: Dict[int, np.ndarray]) -> pd.DataFrame:
    rows = []
    common_ids = sorted(set(ref_trajs.keys()) & set(run_trajs.keys()))
    for aid in common_ids:
        ref_xy = ref_trajs[aid]
        run_xy = run_trajs[aid]
        rmse_2d, rmse_x, rmse_y = rmse_time_normalized(ref_xy, run_xy, samples=200)
        rows.append(
            {
                "id": aid,
                "rmse_2d": rmse_2d,
                "rmse_x": rmse_x,
                "rmse_y": rmse_y,
                "n_ref": len(ref_xy),
                "n_run": len(run_xy),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("id")
    return df


def compute_rmse_overall(per_actor: pd.DataFrame) -> Dict[str, float]:
    if per_actor.empty:
        return {"rmse_2d_mean": float("nan"), "rmse_2d_median": float("nan")}
    vals = per_actor["rmse_2d"].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"rmse_2d_mean": float("nan"), "rmse_2d_median": float("nan")}
    return {
        "rmse_2d_mean": float(np.mean(vals)),
        "rmse_2d_median": float(np.median(vals)),
    }


# ----------------------------
# Timing parsing + plots
# ----------------------------
def load_timing_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = read_csv_flexible(path)
    if df.empty:
        return df
    # Numeric coercion for all columns that look numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    # Try to coerce common time columns
    for c in ("frame", "monotonic", "wall_clock", "tick_wall_dt", "tick_wall_dt_ms", "dt_ms", "elapsed_ms", "process_ms"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def pick_time_axes(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    Returns (t_sec_from0, frame_ms, t_label, y_label)
    """
    if df.empty:
        return np.array([]), np.array([]), "elapsed [s]", "time [ms]"

    # Choose elapsed time axis
    if "monotonic" in df.columns and pd.api.types.is_numeric_dtype(df["monotonic"]):
        t = df["monotonic"].to_numpy(dtype=float)
        t = t - np.nanmin(t)
        t_label = "elapsed [s]"
    elif "wall_clock" in df.columns and pd.api.types.is_numeric_dtype(df["wall_clock"]):
        t = df["wall_clock"].to_numpy(dtype=float)
        t = t - np.nanmin(t)
        t_label = "elapsed [s]"
    else:
        t = np.arange(len(df), dtype=float)
        t_label = "index"

    # Choose per-frame time metric (ms)
    candidates_ms = [
        "frame_processed_ms",
        "process_ms",
        "update_ms",
        "dt_ms",
        "elapsed_ms",
        "frame_ms",
        "tick_wall_dt_ms",
    ]
    for c in candidates_ms:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            y = df[c].to_numpy(dtype=float)
            return t, y, t_label, f"{c} [ms]"

    # tick_wall_dt likely seconds
    if "tick_wall_dt" in df.columns and pd.api.types.is_numeric_dtype(df["tick_wall_dt"]):
        y = df["tick_wall_dt"].to_numpy(dtype=float) * 1000.0
        return t, y, t_label, "tick_wall_dt [ms]"

    # fallback: diff of monotonic
    if "monotonic" in df.columns and pd.api.types.is_numeric_dtype(df["monotonic"]):
        mono = df["monotonic"].to_numpy(dtype=float)
        dm = np.diff(mono, prepend=mono[0])
        y = dm * 1000.0
        return t, y, t_label, "diff(monotonic) [ms]"

    y = np.full_like(t, np.nan)
    return t, y, t_label, "time [ms]"


def timing_stats_ms(frame_ms: np.ndarray) -> Dict[str, float]:
    v = frame_ms[np.isfinite(frame_ms)]
    if len(v) == 0:
        return {
            "mean_ms": float("nan"),
            "p50_ms": float("nan"),
            "p90_ms": float("nan"),
            "p95_ms": float("nan"),
            "max_ms": float("nan"),
            "n": 0.0,
        }
    return {
        "mean_ms": float(np.mean(v)),
        "p50_ms": float(np.percentile(v, 50)),
        "p90_ms": float(np.percentile(v, 90)),
        "p95_ms": float(np.percentile(v, 95)),
        "max_ms": float(np.max(v)),
        "n": float(len(v)),
    }


# ----------------------------
# Plotting
# ----------------------------
def setup_paper_fonts():
    # Reduce findfont spam, and keep fallback order.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    # Prefer Arial or BIZ UDP if available.
    mpl.rcParams["font.family"] = [
        "Arial",
        "BIZ UDP Gothic",
        "BIZ UDPゴシック",
        "IPAexGothic",
        "Yu Gothic",
        "Meiryo",
        "MS Gothic",
        "sans-serif",
    ]


def plot_trajectories_pdf(
    trajs: Dict[int, np.ndarray],
    out_pdf: Path,
    *,
    color: str,
    title: Optional[str],
    paper: bool,
):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    for aid in sorted(trajs.keys()):
        pts = trajs[aid]
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=color,
            linewidth=1.8 if paper else 1.2,
            alpha=0.85 if paper else 0.9,
        )

    if title and not paper:
        ax.set_title(title)

    label_fs = 22 if paper else None
    tick_fs = 20 if paper else None

    ax.set_xlabel("X [m]", fontsize=label_fs)
    ax.set_ylabel("Y [m]", fontsize=label_fs)

    if tick_fs:
        ax.tick_params(labelsize=tick_fs, direction="in")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_timing_pdf(
    t_sec: np.ndarray,
    frame_ms: np.ndarray,
    out_pdf: Path,
    *,
    title: Optional[str],
    paper: bool,
    t_label: str,
    y_label: str,
):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_sec, frame_ms, linewidth=1.6 if paper else 1.2)

    if title and not paper:
        ax.set_title(title)

    label_fs = 22 if paper else None
    tick_fs = 20 if paper else None
    ax.set_xlabel(t_label, fontsize=label_fs)
    ax.set_ylabel(y_label, fontsize=label_fs)

    if tick_fs:
        ax.tick_params(labelsize=tick_fs, direction="in")
    ax.grid(True, linestyle="--", alpha=0.35)

    # Ensure x starts at 0 (user requirement)
    if len(t_sec) > 0 and np.isfinite(t_sec).any():
        ax.set_xlim(left=0.0)

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Sweep discovery
# ----------------------------
RUN_DIR_RE = re.compile(r"^N(?P<N>\d+)_Ts(?P<TS>[0-9.]+)$")


def find_runs(outdir: Path) -> List[Tuple[str, Path, int, str]]:
    runs = []
    for p in outdir.iterdir():
        if not p.is_dir():
            continue
        m = RUN_DIR_RE.match(p.name)
        if not m:
            continue
        n = int(m.group("N"))
        ts = m.group("TS")
        runs.append((p.name, p, n, ts))
    runs.sort(key=lambda x: (x[2], float(x[3])))
    return runs


# ----------------------------
# Main
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True, help="Sweep outdir (contains Nxx_TsYY subdirs)")
    ap.add_argument("--ref", type=Path, required=True, help="Reference CSV (exp_300.csv)")
    ap.add_argument("--analysis-name", default=None, help="Optional analysis directory name (default: auto timestamp)")
    ap.add_argument("--paper", action="store_true", help="Paper style fonts/sizes; PDF output")
    ap.add_argument("--color", default="#3da5d9", help="Single trajectory color (e.g. light blue)")
    ap.add_argument("--kind", default="vehicle", choices=["vehicle", "walker", "all"], help="Actor kind to plot/analyze")
    ap.add_argument("--rmse-samples", type=int, default=200, help="Interpolation samples for RMSE")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    outdir = args.outdir
    if not outdir.exists():
        print(f"[ERROR] outdir not found: {outdir}")
        return 2

    if args.paper:
        setup_paper_fonts()

    # Create NEW analysis directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_name = args.analysis_name or f"analysis_{ts}"
    analysis_dir = outdir / analysis_name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Read reference
    ref_raw = read_csv_flexible(args.ref)
    ref = normalize_ref(ref_raw)

    kind = args.kind
    kind_for_filter = None if kind == "all" else kind

    ref_trajs_all = trajectories_by_id(ref, kind_for_filter or "vehicle")

    # Generate reference plots for N=10..100 step10 (separate, no overlay)
    ref_out = analysis_dir / "ref_trajectories"
    ref_out.mkdir(parents=True, exist_ok=True)

    # Determine ref ids order (sorted by id)
    ref_ids_sorted = sorted(ref_trajs_all.keys())

    for n in range(10, 101, 10):
        picked_ids = [i for i in ref_ids_sorted if i <= n]  # ids are 1..N in your data
        trajs_n = {i: ref_trajs_all[i] for i in picked_ids if i in ref_trajs_all}
        if len(trajs_n) == 0:
            continue
        plot_trajectories_pdf(
            trajs_n,
            ref_out / f"ref_trajectories_N{n}.pdf",
            color=args.color,
            title=f"Reference trajectories (N={n})",
            paper=args.paper,
        )

    # Runs
    runs = find_runs(outdir)
    if not runs:
        print(f"[ERROR] No run dirs like N10_Ts0.10 found under: {outdir}")
        return 2

    summary_rows = []

    for tag, run_path, n, ts_str in runs:
        replay_csv = run_path / "replay_state.csv"
        timing_csv = run_path / "stream_timing.csv"

        if not replay_csv.exists():
            print(f"[WARN] missing replay_state.csv: {replay_csv}")
            continue

        run_raw = read_csv_flexible(replay_csv)
        run_df = normalize_run(run_raw)
        run_trajs = trajectories_by_id(run_df, kind_for_filter or "vehicle")

        # Reference subset for this N (ids 1..N)
        ref_sub_ids = [i for i in ref_ids_sorted if i <= n]
        ref_sub_trajs = {i: ref_trajs_all[i] for i in ref_sub_ids if i in ref_trajs_all}

        # RMSE per actor (id match, time-normalized)
        per_actor = compute_rmse_per_actor(ref_sub_trajs, run_trajs)
        rmse_overall = compute_rmse_overall(per_actor)

        run_out = analysis_dir / "runs" / tag
        run_out.mkdir(parents=True, exist_ok=True)

        per_actor_path = run_out / f"per_actor_rmse_{tag}.csv"
        per_actor.to_csv(per_actor_path, index=False)

        # Trajectory plot (run only)
        traj_pdf = run_out / f"trajectories_{tag}.pdf"
        plot_trajectories_pdf(
            run_trajs,
            traj_pdf,
            color=args.color,
            title=f"Run trajectories {tag}",
            paper=args.paper,
        )

        # Timing plot + stats
        tdf = load_timing_csv(timing_csv)
        t_sec, frame_ms, t_label, y_label = pick_time_axes(tdf)
        stats = timing_stats_ms(frame_ms)

        timing_pdf = run_out / f"timing_{tag}.pdf"
        plot_timing_pdf(
            t_sec,
            frame_ms,
            timing_pdf,
            title=f"Timing {tag}",
            paper=args.paper,
            t_label=t_label,
            y_label=y_label,
        )

        summary_rows.append(
            {
                "tag": tag,
                "N": n,
                "Ts": float(ts_str),
                "rmse_2d_mean": rmse_overall["rmse_2d_mean"],
                "rmse_2d_median": rmse_overall["rmse_2d_median"],
                "timing_mean_ms": stats["mean_ms"],
                "timing_p50_ms": stats["p50_ms"],
                "timing_p90_ms": stats["p90_ms"],
                "timing_p95_ms": stats["p95_ms"],
                "timing_max_ms": stats["max_ms"],
                "timing_n": stats["n"],
                "per_actor_csv": str(per_actor_path.relative_to(analysis_dir)),
                "traj_pdf": str(traj_pdf.relative_to(analysis_dir)),
                "timing_pdf": str(timing_pdf.relative_to(analysis_dir)),
            }
        )

        print(f"[OK] {tag}: wrote trajectories/timing/RMSE into {run_out}")

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["N", "Ts"])
    summary_path = analysis_dir / "summary_runs.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"[OK] Analysis dir: {analysis_dir}")
    print(f"[OK] Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
