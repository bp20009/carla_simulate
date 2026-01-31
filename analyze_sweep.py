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

RMSE implementation (fixed vs your current):
  - Uses a *time axis* per actor (run: monotonic_time by default, ref: frame*Ts)
  - Compares only a time window with the same duration.
  - Optionally searches for the best time offset of the run window inside the ref trajectory
    (this solves "frame doesn't match" and also solves "run is a short segment of a long ref").
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
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
    df = df.dropna(subset=["id", "x", "y", "frame"])
    df["id"] = df["id"].astype(int)
    df["type"] = df["type"].astype(str)
    df = df.sort_values(["id", "frame"])
    return df


def normalize_run(df: pd.DataFrame) -> pd.DataFrame:
    """
    replay_state.csv is produced by vehicle_state_stream.py (or similar).
    We keep useful time columns such as monotonic_time/wall_time if present.
    """
    colmap = {}

    # id (your file has "id" as stable internal id, and also carla_actor_id)
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

    # frame (CARLA frame index, not necessarily matching ref)
    if "frame" in df.columns:
        colmap["frame"] = "frame"
    elif "timestamp" in df.columns:
        colmap["timestamp"] = "frame"

    # time axis candidates
    if "monotonic_time" in df.columns:
        colmap["monotonic_time"] = "monotonic_time"
    if "wall_time" in df.columns:
        colmap["wall_time"] = "wall_time"
    if "frame_elapsed" in df.columns:
        colmap["frame_elapsed"] = "frame_elapsed"
    if "tick_wall_dt" in df.columns:
        colmap["tick_wall_dt"] = "tick_wall_dt"

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

    num_cols = ["id", "x", "y"]
    for c in ("z", "frame", "monotonic_time", "wall_time", "frame_elapsed", "tick_wall_dt"):
        if c in df.columns:
            num_cols.append(c)
    df = to_numeric_df(df, num_cols)
    df = df.dropna(subset=["id", "x", "y"])
    df["id"] = df["id"].astype(int)

    if "type" in df.columns:
        df["type"] = df["type"].astype(str)
    else:
        df["type"] = "unknown"

    # sort by best available time axis for stability
    sort_cols = ["id"]
    if "monotonic_time" in df.columns and df["monotonic_time"].notna().any():
        sort_cols.append("monotonic_time")
    elif "frame" in df.columns and df["frame"].notna().any():
        sort_cols.append("frame")
    df = df.sort_values(sort_cols)
    return df


# ----------------------------
# Trajectory extraction
# ----------------------------
def _filter_kind(df: pd.DataFrame, kind: Optional[str]) -> pd.DataFrame:
    if not kind or kind == "all":
        return df
    if "type" not in df.columns:
        return df
    return df[df["type"].astype(str).str.startswith(kind)]


def trajectories_xy_by_id(df: pd.DataFrame, kind: Optional[str] = "vehicle") -> Dict[int, np.ndarray]:
    """
    For plotting.
    Return dict: id -> array[[x,y], ...]
    """
    sub = _filter_kind(df, kind)
    trajs: Dict[int, np.ndarray] = {}
    for aid, g in sub.groupby("id", sort=True):
        pts = g[["x", "y"]].to_numpy(dtype=float)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) >= 2:
            trajs[int(aid)] = pts
    return trajs


def choose_time_column(df: pd.DataFrame, preferred: str, fallbacks: List[str]) -> Optional[str]:
    """
    Pick a usable time column from df.
    A usable column must exist and contain at least 2 finite values.
    """
    candidates = [preferred] + fallbacks
    for c in candidates:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if len(v) >= 2:
                return c
    return None


def trajectories_time_xy_by_id(
    df: pd.DataFrame,
    *,
    kind: Optional[str],
    time_col: str,
    time_scale: float = 1.0,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    For RMSE.
    Return dict: id -> (t_sec, xy)
    where t_sec starts at 0 for each actor and is strictly non-decreasing.
    """
    sub = _filter_kind(df, kind)

    trajs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for aid, g in sub.groupby("id", sort=True):
        if time_col not in g.columns:
            continue
        t = pd.to_numeric(g[time_col], errors="coerce").to_numpy(dtype=float) * float(time_scale)
        xy = g[["x", "y"]].to_numpy(dtype=float)

        ok = np.isfinite(t) & np.isfinite(xy).all(axis=1)
        t = t[ok]
        xy = xy[ok]
        if len(t) < 2:
            continue

        # sort by t
        order = np.argsort(t)
        t = t[order]
        xy = xy[order]

        # make relative time
        t = t - t[0]

        # enforce non-decreasing (dedup equal times)
        keep = np.ones_like(t, dtype=bool)
        keep[1:] = t[1:] >= t[:-1]
        t = t[keep]
        xy = xy[keep]

        # if many identical times remain, compress to unique times by taking last sample
        if len(t) >= 2:
            # unique with last occurrence
            _, idx = np.unique(t, return_index=True)
            # idx are first occurrences; we want last occurrences:
            # invert trick by unique on reversed
            t_rev = t[::-1]
            _, idx_rev = np.unique(t_rev, return_index=True)
            last_idx = (len(t) - 1) - idx_rev
            last_idx = np.sort(last_idx)
            t = t[last_idx]
            xy = xy[last_idx]

        if len(t) >= 2:
            trajs[int(aid)] = (t, xy)

    return trajs


# ----------------------------
# RMSE (id match, time-aligned)
# ----------------------------
def rmse_on_grid(
    ref_t: np.ndarray,
    ref_xy: np.ndarray,
    run_t: np.ndarray,
    run_xy: np.ndarray,
    *,
    t0_ref: float,
    duration: float,
    samples: int,
) -> Tuple[float, float, float]:
    """
    Compute RMSE over [0, duration] using:
      ref(t0_ref + t) vs run(t)
    Both are interpolated onto a common grid.
    """
    if duration <= 0 or samples < 2:
        return (float("nan"), float("nan"), float("nan"))

    t_grid = np.linspace(0.0, duration, samples)

    ref_x = np.interp(t0_ref + t_grid, ref_t, ref_xy[:, 0])
    ref_y = np.interp(t0_ref + t_grid, ref_t, ref_xy[:, 1])
    run_x = np.interp(t_grid, run_t, run_xy[:, 0])
    run_y = np.interp(t_grid, run_t, run_xy[:, 1])

    dx = run_x - ref_x
    dy = run_y - ref_y

    rmse_x = float(np.sqrt(np.mean(dx * dx)))
    rmse_y = float(np.sqrt(np.mean(dy * dy)))
    rmse_2d = float(np.sqrt(np.mean(dx * dx + dy * dy)))
    return rmse_2d, rmse_x, rmse_y


def rmse_time_best_offset(
    ref_t: np.ndarray,
    ref_xy: np.ndarray,
    run_t: np.ndarray,
    run_xy: np.ndarray,
    *,
    samples: int,
    max_candidates: int,
    offset_step_sec: float,
) -> Tuple[float, float, float, float]:
    """
    Compare a run trajectory window [0, T_run] against the best matching window in ref
    by searching an offset t0_ref in [0, T_ref - T_run].

    Returns (rmse_2d, rmse_x, rmse_y, best_offset_sec)
    """
    if len(ref_t) < 2 or len(run_t) < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    T_run = float(run_t.max())
    T_ref = float(ref_t.max())
    if not np.isfinite(T_run) or not np.isfinite(T_ref) or T_run <= 0 or T_ref <= 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    duration = min(T_run, T_ref)
    if duration <= 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    # If ref is not longer than run, just align starts (offset 0).
    max_offset = max(0.0, T_ref - duration)
    if max_offset <= 0.0:
        rmse_2d, rmse_x, rmse_y = rmse_on_grid(ref_t, ref_xy, run_t, run_xy, t0_ref=0.0, duration=duration, samples=samples)
        return rmse_2d, rmse_x, rmse_y, 0.0

    # Build offset candidates (bounded count)
    if offset_step_sec <= 0:
        offset_step_sec = max_offset / max(1, (max_candidates - 1))

    cand = np.arange(0.0, max_offset + 1e-9, offset_step_sec, dtype=float)
    if len(cand) > max_candidates:
        # subsample evenly to cap
        idx = np.linspace(0, len(cand) - 1, max_candidates).round().astype(int)
        cand = cand[idx]

    best = (float("inf"), float("nan"), float("nan"), float("nan"))  # rmse2d, rmsex, rmsey, offset
    for t0 in cand:
        r2, rx, ry = rmse_on_grid(ref_t, ref_xy, run_t, run_xy, t0_ref=float(t0), duration=duration, samples=samples)
        if np.isfinite(r2) and r2 < best[0]:
            best = (r2, rx, ry, float(t0))

    if not np.isfinite(best[0]):
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    return best[0], best[1], best[2], best[3]


def compute_rmse_per_actor(
    ref_trajs: Dict[int, Tuple[np.ndarray, np.ndarray]],
    run_trajs: Dict[int, Tuple[np.ndarray, np.ndarray]],
    *,
    samples: int,
    max_candidates: int,
    offset_step_sec: float,
) -> pd.DataFrame:
    rows = []
    common_ids = sorted(set(ref_trajs.keys()) & set(run_trajs.keys()))
    for aid in common_ids:
        ref_t, ref_xy = ref_trajs[aid]
        run_t, run_xy = run_trajs[aid]
        rmse_2d, rmse_x, rmse_y, best_offset = rmse_time_best_offset(
            ref_t,
            ref_xy,
            run_t,
            run_xy,
            samples=samples,
            max_candidates=max_candidates,
            offset_step_sec=offset_step_sec,
        )
        rows.append(
            {
                "id": aid,
                "rmse_2d": rmse_2d,
                "rmse_x": rmse_x,
                "rmse_y": rmse_y,
                "best_offset_sec": best_offset,
                "t_ref_end_sec": float(ref_t.max()) if len(ref_t) else float("nan"),
                "t_run_end_sec": float(run_t.max()) if len(run_t) else float("nan"),
                "n_ref": int(len(ref_xy)),
                "n_run": int(len(run_xy)),
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
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    for c in ("frame", "monotonic", "wall_clock", "tick_wall_dt", "tick_wall_dt_ms", "dt_ms", "elapsed_ms", "process_ms"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def pick_time_axes(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, str, str]:
    if df.empty:
        return np.array([]), np.array([]), "elapsed [s]", "time [ms]"

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

    if "tick_wall_dt" in df.columns and pd.api.types.is_numeric_dtype(df["tick_wall_dt"]):
        y = df["tick_wall_dt"].to_numpy(dtype=float) * 1000.0
        return t, y, t_label, "tick_wall_dt [ms]"

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
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
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

    # RMSE controls
    ap.add_argument("--rmse-samples", type=int, default=200, help="Interpolation samples for RMSE")
    ap.add_argument(
        "--run-time-col",
        default="monotonic_time",
        help="Preferred time column for run (default: monotonic_time). Fallbacks: wall_time, frame.",
    )
    ap.add_argument(
        "--ref-time-col",
        default="frame",
        help="Time column for ref (default: frame). Usually frame.",
    )
    ap.add_argument(
        "--ref-time-scale",
        type=float,
        default=None,
        help="Scale applied to ref time column to get seconds. If omitted, uses Ts from run dir name.",
    )
    ap.add_argument(
        "--rmse-offset-step",
        type=float,
        default=0.0,
        help="Search step [sec] for best offset. 0 means auto.",
    )
    ap.add_argument(
        "--rmse-max-candidates",
        type=int,
        default=200,
        help="Max number of offset candidates per actor (cap for speed).",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()

    outdir = args.outdir
    if not outdir.exists():
        print(f"[ERROR] outdir not found: {outdir}")
        return 2

    if args.paper:
        setup_paper_fonts()

    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_name = args.analysis_name or f"analysis_{ts_now}"
    analysis_dir = outdir / analysis_name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Read reference
    ref_raw = read_csv_flexible(args.ref)
    ref_df = normalize_ref(ref_raw)

    kind = args.kind
    kind_for_filter: Optional[str] = None if kind == "all" else kind

    # For reference trajectory plots (xy only)
    ref_xy_all = trajectories_xy_by_id(ref_df, kind_for_filter or "vehicle")
    ref_ids_sorted = sorted(ref_xy_all.keys())

    # Reference plots per N (separate, no overlay)
    ref_out = analysis_dir / "ref_trajectories"
    ref_out.mkdir(parents=True, exist_ok=True)

    for n in range(10, 101, 10):
        picked_ids = [i for i in ref_ids_sorted if i <= n]  # assumes ids 1..N
        trajs_n = {i: ref_xy_all[i] for i in picked_ids if i in ref_xy_all}
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

        # Plot trajectories (run only)
        run_xy = trajectories_xy_by_id(run_df, kind_for_filter or "vehicle")
        run_out = analysis_dir / "runs" / tag
        run_out.mkdir(parents=True, exist_ok=True)

        traj_pdf = run_out / f"trajectories_{tag}.pdf"
        plot_trajectories_pdf(
            run_xy,
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

        # ----------------------------
        # RMSE (time-aligned, best offset)
        # ----------------------------
        Ts_run = float(ts_str)  # from directory name Nxx_TsYY

        # choose run time column
        run_time_col = choose_time_column(
            run_df,
            preferred=args.run_time_col,
            fallbacks=["wall_time", "frame"],
        )
        if run_time_col is None:
            print(f"[WARN] {tag}: no usable run time column among {args.run_time_col}/wall_time/frame; skip RMSE")
            per_actor = pd.DataFrame()
            rmse_overall = {"rmse_2d_mean": float("nan"), "rmse_2d_median": float("nan")}
        else:
            # build ref/run time trajectories
            # ref time axis: usually frame * Ts_run (unless overridden)
            ref_time_col = args.ref_time_col
            if ref_time_col not in ref_df.columns:
                raise ValueError(f"ref missing ref_time_col={ref_time_col}. columns={list(ref_df.columns)}")

            ref_time_scale = float(args.ref_time_scale) if args.ref_time_scale is not None else Ts_run
            run_time_scale = 1.0
            # If run_time_col is "frame", interpret like frame*Ts_run to get seconds.
            if run_time_col == "frame":
                run_time_scale = Ts_run

            ref_time_trajs_all = trajectories_time_xy_by_id(
                ref_df,
                kind=kind_for_filter or "vehicle",
                time_col=ref_time_col,
                time_scale=ref_time_scale,
            )
            run_time_trajs = trajectories_time_xy_by_id(
                run_df,
                kind=kind_for_filter or "vehicle",
                time_col=run_time_col,
                time_scale=run_time_scale,
            )

            # Reference subset for this N (ids 1..N)
            ref_sub_ids = [i for i in sorted(ref_time_trajs_all.keys()) if i <= n]
            ref_sub_trajs = {i: ref_time_trajs_all[i] for i in ref_sub_ids if i in ref_time_trajs_all}

            # best-offset RMSE
            offset_step = float(args.rmse_offset_step)
            if offset_step <= 0:
                # auto: roughly Ts_run, but not too small
                offset_step = max(Ts_run, 0.01)

            per_actor = compute_rmse_per_actor(
                ref_sub_trajs,
                run_time_trajs,
                samples=int(args.rmse_samples),
                max_candidates=int(args.rmse_max_candidates),
                offset_step_sec=offset_step,
            )
            rmse_overall = compute_rmse_overall(per_actor)

        per_actor_path = run_out / f"per_actor_rmse_{tag}.csv"
        per_actor.to_csv(per_actor_path, index=False)

        # helpful debug counts
        common_ids = sorted(set(ref_ids_sorted) & set(run_xy.keys()))
        summary_rows.append(
            {
                "tag": tag,
                "N": n,
                "Ts": float(ts_str),
                "run_time_col": run_time_col or "",
                "ref_time_scale_sec_per_unit": float(args.ref_time_scale) if args.ref_time_scale is not None else Ts_run,
                "rmse_2d_mean": rmse_overall["rmse_2d_mean"],
                "rmse_2d_median": rmse_overall["rmse_2d_median"],
                "common_ids_xy_count": int(len(common_ids)),
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
