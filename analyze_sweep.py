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

Outputs under a NEW analysis directory:
  - per-run trajectory PDFs
  - reference trajectory PDFs per N (separate)
  - per-run timing plots and summary stats
  - per-run per-actor metrics (RMSE + NN + coverage)

Key fix:
  - Actor matching uses --match-key (default: external_id), NOT "id".
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# ----------------------------
# CSV helpers
# ----------------------------
def _strip_null_bytes(lines: Iterable[str]) -> Iterator[str]:
    for line in lines:
        yield line.replace("\x00", "")


def read_csv_flexible(path: Path) -> pd.DataFrame:
    """
    Robust CSV/TSV reader:
      - tries common encodings
      - uses pandas separator auto-detection (sep=None, engine='python')
      - strips null bytes if needed
    """
    encodings = ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp932")
    last_exc: Optional[Exception] = None
    for enc in encodings:
        try:
            # Use pandas sniffing for separators (handles CSV/TSV)
            df = pd.read_csv(
                path,
                encoding=enc,
                sep=None,
                engine="python",
            )
            # Strip null bytes in column names if any
            df.columns = [c.replace("\x00", "") for c in df.columns]
            return df
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to read CSV/TSV with fallback encodings: {path}") from last_exc


def to_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ----------------------------
# Normalization
# ----------------------------
def normalize_ref(df: pd.DataFrame) -> pd.DataFrame:
    required_any = {"frame", "id", "type"}
    if not required_any.issubset(set(df.columns)):
        raise ValueError(f"ref missing required columns {required_any}. got={list(df.columns)}")

    if "x" not in df.columns and "location_x" in df.columns:
        df = df.rename(columns={"location_x": "x"})
    if "y" not in df.columns and "location_y" in df.columns:
        df = df.rename(columns={"location_y": "y"})
    if "z" not in df.columns and "location_z" in df.columns:
        df = df.rename(columns={"location_z": "z"})

    df = to_numeric_df(df, ["frame", "id", "x", "y", "z"])
    df = df.dropna(subset=["frame", "id", "x", "y"])
    df["id"] = df["id"].astype(int)
    df["type"] = df["type"].astype(str)
    df = df.sort_values(["id", "frame"])

    # unified match id
    df["match_id"] = df["id"].astype(int)
    return df


def normalize_run(df: pd.DataFrame) -> pd.DataFrame:
    # map expected columns
    colmap = {}

    # stable internal id
    if "id" in df.columns:
        colmap["id"] = "id"

    # match candidates
    for c in ("external_id", "object_id", "carla_actor_id"):
        if c in df.columns:
            colmap[c] = c

    # type
    if "type" in df.columns:
        colmap["type"] = "type"
    elif "actor_type" in df.columns:
        colmap["actor_type"] = "type"

    # frame
    if "frame" in df.columns:
        colmap["frame"] = "frame"
    elif "timestamp" in df.columns:
        colmap["timestamp"] = "frame"

    # time candidates
    for c in ("monotonic_time", "wall_time", "frame_elapsed", "tick_wall_dt"):
        if c in df.columns:
            colmap[c] = c

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

    required = {"x", "y"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"run CSV missing required columns {required}. got={list(df.columns)}")

    num_cols = ["x", "y"]
    for c in ("z", "frame", "id", "external_id", "object_id", "carla_actor_id", "monotonic_time", "wall_time", "frame_elapsed", "tick_wall_dt"):
        if c in df.columns:
            num_cols.append(c)

    df = to_numeric_df(df, num_cols)
    df = df.dropna(subset=["x", "y"])

    if "type" in df.columns:
        df["type"] = df["type"].astype(str)
    else:
        df["type"] = "unknown"

    # keep integer-ish ids
    for c in ("id", "external_id", "object_id", "carla_actor_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ----------------------------
# Kind filter
# ----------------------------
def _filter_kind(df: pd.DataFrame, kind: Optional[str]) -> pd.DataFrame:
    if not kind or kind == "all":
        return df
    if "type" not in df.columns:
        return df
    return df[df["type"].astype(str).str.startswith(kind)]


# ----------------------------
# Matching + trajectory extraction
# ----------------------------
def apply_match_id_run(df: pd.DataFrame, match_key: str) -> pd.DataFrame:
    if match_key == "id":
        if "id" not in df.columns:
            raise ValueError("run missing id column for match-key=id")
        df["match_id"] = df["id"]
    else:
        if match_key not in df.columns:
            raise ValueError(f"run missing match key column: {match_key}")
        df["match_id"] = df[match_key]

    df = df.dropna(subset=["match_id"])
    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce")
    df = df.dropna(subset=["match_id"])
    df["match_id"] = df["match_id"].astype(int)
    return df


def trajectories_xy_by_match_id(df: pd.DataFrame, kind: Optional[str]) -> Dict[int, np.ndarray]:
    sub = _filter_kind(df, kind)
    if "match_id" not in sub.columns:
        raise ValueError("missing match_id in df (did you call apply_match_id_run?)")

    trajs: Dict[int, np.ndarray] = {}
    for mid, g in sub.groupby("match_id", sort=True):
        pts = g[["x", "y"]].to_numpy(dtype=float)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) >= 2:
            trajs[int(mid)] = pts
    return trajs


def choose_time_column(df: pd.DataFrame, preferred: str, fallbacks: List[str]) -> Optional[str]:
    candidates = [preferred] + fallbacks
    for c in candidates:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if len(v) >= 2:
                return c
    return None


def trajectories_time_xy_by_match_id(
    df: pd.DataFrame,
    *,
    kind: Optional[str],
    time_col: str,
    time_scale: float = 1.0,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    sub = _filter_kind(df, kind)
    if "match_id" not in sub.columns:
        raise ValueError("missing match_id in df")

    trajs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for mid, g in sub.groupby("match_id", sort=True):
        if time_col not in g.columns:
            continue
        t = pd.to_numeric(g[time_col], errors="coerce").to_numpy(dtype=float) * float(time_scale)
        xy = g[["x", "y"]].to_numpy(dtype=float)

        ok = np.isfinite(t) & np.isfinite(xy).all(axis=1)
        t = t[ok]
        xy = xy[ok]
        if len(t) < 2:
            continue

        order = np.argsort(t)
        t = t[order]
        xy = xy[order]

        t = t - t[0]

        # ensure non-decreasing
        keep = np.ones_like(t, dtype=bool)
        keep[1:] = t[1:] >= t[:-1]
        t = t[keep]
        xy = xy[keep]

        # unique times (keep last occurrence)
        if len(t) >= 2:
            t_rev = t[::-1]
            _, idx_rev = np.unique(t_rev, return_index=True)
            last_idx = (len(t) - 1) - idx_rev
            last_idx = np.sort(last_idx)
            t = t[last_idx]
            xy = xy[last_idx]

        if len(t) >= 2:
            trajs[int(mid)] = (t, xy)

    return trajs


# ----------------------------
# RMSE (time best offset)
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
) -> Tuple[float, float, float, float, float, float, float]:
    if len(ref_t) < 2 or len(run_t) < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    T_run = float(run_t.max())
    T_ref = float(ref_t.max())
    if not np.isfinite(T_run) or not np.isfinite(T_ref) or T_run <= 0 or T_ref <= 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), T_ref, T_run)

    duration = min(T_run, T_ref)
    if duration <= 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), duration, T_ref, T_run)

    max_offset = max(0.0, T_ref - duration)
    if max_offset <= 0.0:
        r2, rx, ry = rmse_on_grid(ref_t, ref_xy, run_t, run_xy, t0_ref=0.0, duration=duration, samples=samples)
        return r2, rx, ry, 0.0, duration, T_ref, T_run

    if offset_step_sec <= 0:
        offset_step_sec = max_offset / max(1, (max_candidates - 1))

    cand = np.arange(0.0, max_offset + 1e-9, offset_step_sec, dtype=float)
    if len(cand) > max_candidates:
        idx = np.linspace(0, len(cand) - 1, max_candidates).round().astype(int)
        cand = cand[idx]

    best = (float("inf"), float("nan"), float("nan"), float("nan"))
    best_t0 = float("nan")
    for t0 in cand:
        r2, rx, ry = rmse_on_grid(ref_t, ref_xy, run_t, run_xy, t0_ref=float(t0), duration=duration, samples=samples)
        if np.isfinite(r2) and r2 < best[0]:
            best = (r2, rx, ry, float(t0))
            best_t0 = float(t0)

    if not np.isfinite(best[0]):
        return (float("nan"), float("nan"), float("nan"), float("nan"), duration, T_ref, T_run)
    return best[0], best[1], best[2], best_t0, duration, T_ref, T_run


# ----------------------------
# Nearest-neighbor + coverage metrics
# ----------------------------
def path_length_m(xy: np.ndarray) -> float:
    if xy is None or len(xy) < 2:
        return 0.0
    d = np.diff(xy, axis=0)
    return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))


def nn_dist_series(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    """
    For each point in a_xy, compute min distance to any point in b_xy.
    O(N*M) but ok for typical sizes. If huge, optimize with KDTree.
    """
    if len(a_xy) == 0 or len(b_xy) == 0:
        return np.array([], dtype=float)
    # broadcast: (Na,1,2) - (1,Nb,2) -> (Na,Nb,2)
    diff = a_xy[:, None, :] - b_xy[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return np.min(dist, axis=1)


def summarize_nn(dist: np.ndarray) -> Dict[str, float]:
    if dist.size == 0:
        return {
            "nn_mean_m": float("nan"),
            "nn_p50_m": float("nan"),
            "nn_p90_m": float("nan"),
            "nn_p95_m": float("nan"),
            "nn_max_m": float("nan"),
            "nn_end_m": float("nan"),
        }
    v = dist[np.isfinite(dist)]
    if v.size == 0:
        return {
            "nn_mean_m": float("nan"),
            "nn_p50_m": float("nan"),
            "nn_p90_m": float("nan"),
            "nn_p95_m": float("nan"),
            "nn_max_m": float("nan"),
            "nn_end_m": float("nan"),
        }
    return {
        "nn_mean_m": float(np.mean(v)),
        "nn_p50_m": float(np.percentile(v, 50)),
        "nn_p90_m": float(np.percentile(v, 90)),
        "nn_p95_m": float(np.percentile(v, 95)),
        "nn_max_m": float(np.max(v)),
        "nn_end_m": float(v[-1]),
    }


def coverage(dist: np.ndarray, tau_m: float) -> float:
    v = dist[np.isfinite(dist)]
    if v.size == 0:
        return float("nan")
    return float(np.mean(v <= float(tau_m)))


# ----------------------------
# Timing
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

    candidates_ms = ["frame_processed_ms", "process_ms", "update_ms", "dt_ms", "elapsed_ms", "frame_ms", "tick_wall_dt_ms"]
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
        return {"mean_ms": float("nan"), "p50_ms": float("nan"), "p90_ms": float("nan"), "p95_ms": float("nan"), "max_ms": float("nan"), "n": 0.0}
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
    mpl.rcParams["font.family"] = ["Arial", "BIZ UDP Gothic", "BIZ UDPゴシック", "IPAexGothic", "Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]


def plot_trajectories_pdf(trajs: Dict[int, np.ndarray], out_pdf: Path, *, color: str, title: Optional[str], paper: bool):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    for aid in sorted(trajs.keys()):
        pts = trajs[aid]
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.8 if paper else 1.2, alpha=0.85 if paper else 0.9)
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


def plot_timing_pdf(t_sec: np.ndarray, frame_ms: np.ndarray, out_pdf: Path, *, title: Optional[str], paper: bool, t_label: str, y_label: str):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_sec, frame_ms, linewidth=1.6 if paper else 1.2)
    if title and not paper:
        ax.set_title(title)

    label_fs = 22 if paper else None
    tick_fs = 20 if paper else None
    ax.set_xlabel("経過フレーム", fontsize=label_fs)
    ax.set_ylabel("処理時間 [ms]", fontsize=label_fs)
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
# CLI
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--analysis-name", default=None)
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--color", default="#3da5d9")
    ap.add_argument("--kind", default="vehicle", choices=["vehicle", "walker", "all"])

    # backward-compat (ignored)
    ap.add_argument("--per-actor", action="store_true", help="(compat) ignored; per-actor CSV is always written")

    # matching
    ap.add_argument(
        "--match-key",
        default="external_id",
        choices=["id", "external_id", "object_id", "carla_actor_id"],
        help="Actor matching key for run. Default: external_id (recommended).",
    )

    # RMSE controls
    ap.add_argument("--rmse-samples", type=int, default=200)
    ap.add_argument("--run-time-col", default="monotonic_time")
    ap.add_argument("--ref-time-col", default="frame")
    ap.add_argument("--ref-time-scale", type=float, default=None)
    ap.add_argument("--rmse-offset-step", type=float, default=0.0)
    ap.add_argument("--rmse-max-candidates", type=int, default=200)

    # NN/Coverage thresholds
    ap.add_argument("--cov-thresholds", default="1,2,3,5", help="comma-separated thresholds in meters for coverage")
    return ap


# ----------------------------
# Main
# ----------------------------
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

    # thresholds
    cov_taus = [float(x) for x in str(args.cov_thresholds).split(",") if x.strip()]

    # Read ref
    ref_df = normalize_ref(read_csv_flexible(args.ref))

    kind = args.kind
    kind_for_filter: Optional[str] = None if kind == "all" else kind

    # ref XY for plotting (match_id already in ref_df)
    ref_xy_all = trajectories_xy_by_match_id(ref_df, kind_for_filter)

    ref_out = analysis_dir / "ref_trajectories"
    ref_out.mkdir(parents=True, exist_ok=True)
    ref_ids_sorted = sorted(ref_xy_all.keys())

    for n in range(10, 101, 10):
        picked = [i for i in ref_ids_sorted if i <= n]
        trajs_n = {i: ref_xy_all[i] for i in picked if i in ref_xy_all}
        if trajs_n:
            plot_trajectories_pdf(trajs_n, ref_out / f"ref_trajectories_N{n}.pdf", color=args.color, title=f"Reference trajectories (N={n})", paper=args.paper)

    # runs
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

        Ts_run = float(ts_str)

        run_df = normalize_run(read_csv_flexible(replay_csv))
        run_df = apply_match_id_run(run_df, args.match_key)

        run_out = analysis_dir / "runs" / tag
        run_out.mkdir(parents=True, exist_ok=True)

        # plot run trajectories
        run_xy = trajectories_xy_by_match_id(run_df, kind_for_filter)
        traj_pdf = run_out / f"trajectories_{tag}.pdf"
        plot_trajectories_pdf(run_xy, traj_pdf, color=args.color, title=f"Run trajectories {tag}", paper=args.paper)

        # timing
        tdf = load_timing_csv(timing_csv)
        t_sec, frame_ms, t_label, y_label = pick_time_axes(tdf)
        stats = timing_stats_ms(frame_ms)
        timing_pdf = run_out / f"timing_{tag}.pdf"
        plot_timing_pdf(t_sec, frame_ms, timing_pdf, title=f"Timing {tag}", paper=args.paper, t_label=t_label, y_label=y_label)

        # RMSE preparation: time cols
        run_time_col = choose_time_column(run_df, preferred=args.run_time_col, fallbacks=["wall_time", "frame"])
        if run_time_col is None:
            per_actor = pd.DataFrame()
            rmse_mean = float("nan")
            rmse_med = float("nan")
        else:
            ref_time_col = args.ref_time_col
            if ref_time_col not in ref_df.columns:
                raise ValueError(f"ref missing ref_time_col={ref_time_col}. columns={list(ref_df.columns)}")

            ref_time_scale = float(args.ref_time_scale) if args.ref_time_scale is not None else Ts_run
            run_time_scale = Ts_run if run_time_col == "frame" else 1.0

            ref_time_trajs_all = trajectories_time_xy_by_match_id(ref_df, kind=kind_for_filter, time_col=ref_time_col, time_scale=ref_time_scale)
            run_time_trajs = trajectories_time_xy_by_match_id(run_df, kind=kind_for_filter, time_col=run_time_col, time_scale=run_time_scale)

            # subset by N (assumes match ids are 1..N in ref)
            ref_sub_ids = [i for i in sorted(ref_time_trajs_all.keys()) if i <= n]
            ref_sub = {i: ref_time_trajs_all[i] for i in ref_sub_ids}

            # RMSE + NN + coverage per actor
            rows = []
            common = sorted(set(ref_sub.keys()) & set(run_time_trajs.keys()))
            offset_step = float(args.rmse_offset_step)
            if offset_step <= 0:
                offset_step = max(Ts_run, 0.01)

            for mid in common:
                ref_t, ref_xy = ref_sub[mid]
                run_t, run_xy_t = run_time_trajs[mid]

                rmse_2d, rmse_x, rmse_y, best_offset, duration, t_ref_end, t_run_end = rmse_time_best_offset(
                    ref_t, ref_xy, run_t, run_xy_t,
                    samples=int(args.rmse_samples),
                    max_candidates=int(args.rmse_max_candidates),
                    offset_step_sec=offset_step,
                )

                # windowed ref for NN/coverage (same grid as rmse evaluation)
                if np.isfinite(best_offset) and np.isfinite(duration) and duration > 0:
                    t_grid = np.linspace(0.0, duration, int(args.rmse_samples))
                    ref_win = np.column_stack([
                        np.interp(best_offset + t_grid, ref_t, ref_xy[:, 0]),
                        np.interp(best_offset + t_grid, ref_t, ref_xy[:, 1]),
                    ])
                    run_win = np.column_stack([
                        np.interp(t_grid, run_t, run_xy_t[:, 0]),
                        np.interp(t_grid, run_t, run_xy_t[:, 1]),
                    ])
                else:
                    ref_win = np.empty((0, 2), dtype=float)
                    run_win = np.empty((0, 2), dtype=float)

                # NN distances both directions
                d_run_to_ref = nn_dist_series(run_win, ref_win)
                d_ref_to_run = nn_dist_series(ref_win, run_win)

                nn_run = summarize_nn(d_run_to_ref)
                nn_ref = summarize_nn(d_ref_to_run)

                row = {
                    "id": int(mid),
                    "rmse_2d": rmse_2d,
                    "rmse_x": rmse_x,
                    "rmse_y": rmse_y,
                    "best_offset_sec": best_offset,
                    "duration_sec": duration,
                    "t_ref_end_sec": t_ref_end,
                    "t_run_end_sec": t_run_end,
                    "n_ref": int(len(ref_xy)),
                    "n_run": int(len(run_xy_t)),
                    "ref_path_len_m": path_length_m(ref_xy),
                    "run_path_len_m": path_length_m(run_xy_t),
                    **nn_run,
                    "nn_ref_mean_m": nn_ref["nn_mean_m"],
                    "nn_ref_p50_m": nn_ref["nn_p50_m"],
                    "nn_ref_p90_m": nn_ref["nn_p90_m"],
                    "nn_ref_p95_m": nn_ref["nn_p95_m"],
                    "nn_ref_max_m": nn_ref["nn_max_m"],
                    "nn_ref_end_m": nn_ref["nn_end_m"],
                }

                # coverage
                for tau in cov_taus:
                    row[f"cov_run_le_{int(tau) if tau.is_integer() else tau}m"] = coverage(d_run_to_ref, tau)
                    row[f"cov_ref_le_{int(tau) if tau.is_integer() else tau}m"] = coverage(d_ref_to_run, tau)

                rows.append(row)

            per_actor = pd.DataFrame(rows).sort_values("id") if rows else pd.DataFrame()
            if per_actor.empty:
                rmse_mean = float("nan")
                rmse_med = float("nan")
            else:
                v = per_actor["rmse_2d"].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                rmse_mean = float(np.mean(v)) if v.size else float("nan")
                rmse_med = float(np.median(v)) if v.size else float("nan")

        per_actor_path = run_out / f"per_actor_metrics_{tag}.csv"
        per_actor.to_csv(per_actor_path, index=False)

        summary_rows.append(
            {
                "tag": tag,
                "N": n,
                "Ts": Ts_run,
                "match_key": args.match_key,
                "run_time_col": run_time_col or "",
                "rmse_2d_mean": rmse_mean,
                "rmse_2d_median": rmse_med,
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

        print(f"[OK] {tag}: wrote outputs into {run_out}")

    summary_df = pd.DataFrame(summary_rows).sort_values(["N", "Ts"]) if summary_rows else pd.DataFrame()
    summary_path = analysis_dir / "summary_runs.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"[OK] Analysis dir: {analysis_dir}")
    print(f"[OK] Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
