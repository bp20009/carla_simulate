#!/usr/bin/env python3
"""
Analyze CARLA sweep results.

Outputs (under --outdir):
- summary_runs.csv : per-run RMSE + timing
- trajectories_<TAG>.pdf : XY trajectories per run (single color, blue-ish)
- per_actor_rmse_<TAG>.csv : (optional) per-actor RMSE per run when --per-actor

Usage:
  python analyze_sweep.py --outdir sweep_results_20260130_18234773 --ref send_data\\exp_300.csv --paper
  python analyze_sweep.py --outdir sweep_results_20260130_18234773 --ref send_data\\exp_300.csv --paper --per-actor
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

# PDF text embedding
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


# ----------------------------
# Robust CSV reading helpers
# ----------------------------
def _strip_null_bytes(lines: Iterable[str]) -> Iterator[str]:
    for line in lines:
        yield line.replace("\x00", "")


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """
    Read CSV with encoding fallbacks and NUL stripping.
    """
    encodings = ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp932")
    last_exc: Optional[Exception] = None

    # Try pandas first
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_exc = exc

    # Fallback: python csv reader with NUL stripping -> DataFrame
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as fh:
                reader = csv.DictReader(_strip_null_bytes(fh))
                rows = list(reader)
            return pd.DataFrame(rows)
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(f"Failed to read CSV: {path} ({last_exc})")


def _to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ----------------------------
# Data model
# ----------------------------
@dataclass
class RunFiles:
    tag: str
    run_dir: Path
    state_csv: Path
    timing_csv: Optional[Path]


RUN_TAG_RE = re.compile(r"(?:^|\\|/)(N(?P<N>\d+)_Ts(?P<Ts>\d+(?:\.\d+)?))$", re.IGNORECASE)


def discover_runs(outdir: Path) -> List[RunFiles]:
    """
    Find per-run replay_state_*.csv under outdir/**/ and infer tag from folder name.
    """
    runs: List[RunFiles] = []
    for state_csv in sorted(outdir.rglob("replay_state_*.csv")):
        run_dir = state_csv.parent
        tag = run_dir.name

        # Find timing file in same dir
        timing = None
        timing_candidates = list(run_dir.glob("stream_timing_*.csv"))
        if timing_candidates:
            timing = sorted(timing_candidates)[0]

        runs.append(RunFiles(tag=tag, run_dir=run_dir, state_csv=state_csv, timing_csv=timing))

    # If none found, also accept generic *.csv in run folders (optional)
    if not runs:
        for state_csv in sorted(outdir.rglob("*state*.csv")):
            run_dir = state_csv.parent
            tag = run_dir.name
            runs.append(RunFiles(tag=tag, run_dir=run_dir, state_csv=state_csv, timing_csv=None))

    # Unique by tag (keep earliest)
    seen = set()
    uniq: List[RunFiles] = []
    for r in runs:
        if r.tag in seen:
            continue
        seen.add(r.tag)
        uniq.append(r)
    return uniq


def parse_tag(tag: str) -> Tuple[Optional[int], Optional[float]]:
    m = RUN_TAG_RE.search(tag)
    if not m:
        return None, None
    n = int(m.group("N"))
    ts = float(m.group("Ts"))
    return n, ts


# ----------------------------
# Normalization for RMSE
# ----------------------------
def normalize_ref(ref_path: Path) -> pd.DataFrame:
    """
    Load reference (sender input) CSV and return normalized columns:
      actor_id, frame_key, x_ref, y_ref
    """
    df = _read_csv_flexible(ref_path)

    frame_col = _first_existing(df, ["payload_frame", "frame", "pf", "step"])
    id_col = _first_existing(df, ["object_id", "actor_id", "id", "track_id"])
    x_col = _first_existing(df, ["location_x", "x", "pos_x"])
    y_col = _first_existing(df, ["location_y", "y", "pos_y"])

    if frame_col is None or id_col is None or x_col is None or y_col is None:
        raise ValueError(
            f"ref CSV columns not recognized.\n"
            f"Needed frame/id/x/y. Found: {list(df.columns)}"
        )

    _to_numeric(df, [frame_col, id_col, x_col, y_col])

    out = pd.DataFrame()
    out["actor_id"] = df[id_col].astype("Int64")
    out["frame_key"] = df[frame_col].round().astype("Int64")
    out["x_ref"] = df[x_col].astype(float)
    out["y_ref"] = df[y_col].astype(float)

    out = out.dropna(subset=["actor_id", "frame_key", "x_ref", "y_ref"])
    out["actor_id"] = out["actor_id"].astype(int)
    out["frame_key"] = out["frame_key"].astype(int)
    return out


def normalize_run(run_state_csv: Path) -> pd.DataFrame:
    """
    Load run state CSV (vehicle_state_stream output) and return normalized columns:
      actor_id, frame_key, x_run, y_run
    """
    df = _read_csv_flexible(run_state_csv)

    frame_col = _first_existing(df, ["payload_frame", "frame", "pf", "step"])
    # Prefer original object id if present (you used --include-object-id)
    id_col = _first_existing(df, ["object_id", "actor_id", "carla_actor_id", "id", "track_id"])
    x_col = _first_existing(df, ["location_x", "x", "pos_x"])
    y_col = _first_existing(df, ["location_y", "y", "pos_y"])

    if frame_col is None or id_col is None or x_col is None or y_col is None:
        raise ValueError(
            f"run CSV columns not recognized: {run_state_csv}\n"
            f"Needed frame/id/x/y. Found: {list(df.columns)}"
        )

    _to_numeric(df, [frame_col, id_col, x_col, y_col])

    out = pd.DataFrame()
    out["actor_id"] = df[id_col].astype("Int64")
    out["frame_key"] = df[frame_col].round().astype("Int64")
    out["x_run"] = df[x_col].astype(float)
    out["y_run"] = df[y_col].astype(float)

    out = out.dropna(subset=["actor_id", "frame_key", "x_run", "y_run"])
    out["actor_id"] = out["actor_id"].astype(int)
    out["frame_key"] = out["frame_key"].astype(int)
    return out


def compute_rmse(ref_norm: pd.DataFrame, run_norm: pd.DataFrame) -> Tuple[float, float, float, int]:
    """
    Overall RMSE for matched (actor_id, frame_key).
    Returns: (rmse_xy, rmse_x, rmse_y, n_matched)
    """
    merged = pd.merge(ref_norm, run_norm, on=["actor_id", "frame_key"], how="inner")
    if merged.empty:
        return float("nan"), float("nan"), float("nan"), 0

    dx = merged["x_run"].to_numpy() - merged["x_ref"].to_numpy()
    dy = merged["y_run"].to_numpy() - merged["y_ref"].to_numpy()

    mse_xy = np.mean(dx * dx + dy * dy)
    rmse_xy = float(np.sqrt(mse_xy))
    rmse_x = float(np.sqrt(np.mean(dx * dx)))
    rmse_y = float(np.sqrt(np.mean(dy * dy)))
    return rmse_xy, rmse_x, rmse_y, int(len(merged))


def compute_rmse_per_actor(ref_norm: pd.DataFrame, run_norm: pd.DataFrame) -> pd.DataFrame:
    """
    Per-actor RMSE for matched frames.
    """
    merged = pd.merge(ref_norm, run_norm, on=["actor_id", "frame_key"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["actor_id", "n", "rmse_xy", "rmse_x", "rmse_y"])

    merged["dx"] = merged["x_run"] - merged["x_ref"]
    merged["dy"] = merged["y_run"] - merged["y_ref"]

    def _rmse_xy(g: pd.DataFrame) -> float:
        return float(np.sqrt(np.mean(g["dx"].to_numpy() ** 2 + g["dy"].to_numpy() ** 2)))

    def _rmse_1d(arr: np.ndarray) -> float:
        return float(np.sqrt(np.mean(arr ** 2)))

    rows = []
    for aid, g in merged.groupby("actor_id"):
        dx = g["dx"].to_numpy()
        dy = g["dy"].to_numpy()
        rows.append(
            {
                "actor_id": int(aid),
                "n": int(len(g)),
                "rmse_xy": _rmse_xy(g),
                "rmse_x": _rmse_1d(dx),
                "rmse_y": _rmse_1d(dy),
            }
        )
    out = pd.DataFrame(rows).sort_values(["rmse_xy", "n"], ascending=[False, False]).reset_index(drop=True)
    return out


# ----------------------------
# Timing extraction
# ----------------------------
FRAME_MS_RE = re.compile(r"Frame processed in\s+([0-9.]+)\s*ms", re.IGNORECASE)


def timing_from_stream_csv(timing_csv: Path) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    """
    Return (mean_ms, p50_ms, p95_ms, n) from stream_timing CSV.
    Tries to detect a reasonable ms column.
    """
    df = _read_csv_flexible(timing_csv)
    # Convert numeric candidates
    for c in df.columns:
        if "ms" in c.lower() or "dt" in c.lower() or "time" in c.lower():
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Prefer likely columns
    prefer = ["tick_wall_dt_ms", "loop_ms", "frame_ms", "dt_ms", "elapsed_ms"]
    ms_col = _first_existing(df, prefer)
    if ms_col is None:
        # fallback: any column containing 'ms' with numeric content
        ms_candidates = [c for c in df.columns if "ms" in c.lower()]
        for c in ms_candidates:
            if df[c].notna().sum() > 0:
                ms_col = c
                break

    if ms_col is None or df[ms_col].dropna().empty:
        return None, None, None, 0

    v = df[ms_col].dropna().to_numpy(dtype=float)
    mean = float(np.mean(v))
    p50 = float(np.percentile(v, 50))
    p95 = float(np.percentile(v, 95))
    return mean, p50, p95, int(len(v))


def timing_from_receiver_stderr(receiver_stderr: Path) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    """
    Parse 'Frame processed in XX ms' lines.
    """
    if not receiver_stderr.exists():
        return None, None, None, 0

    vals: List[float] = []
    try:
        txt = receiver_stderr.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        txt = receiver_stderr.read_text(encoding="cp932", errors="ignore")

    for m in FRAME_MS_RE.finditer(txt):
        vals.append(float(m.group(1)))

    if not vals:
        return None, None, None, 0

    arr = np.asarray(vals, dtype=float)
    return float(np.mean(arr)), float(np.percentile(arr, 50)), float(np.percentile(arr, 95)), int(len(arr))


# ----------------------------
# Plotting
# ----------------------------
def setup_paper_fonts() -> None:
    # Prefer Arial (Windows default), then your preferred JP fonts if installed
    mpl.rcParams["font.family"] = [
        "Arial",
        "BIZ UDP Gothic",
        "BIZ UDPゴシック",
        "IPAexGothic",
        "MS Gothic",
        "sans-serif",
    ]


def plot_run_trajectories_pdf(
    out_pdf: Path,
    tag: str,
    ref_norm: pd.DataFrame,
    run_norm: pd.DataFrame,
    paper: bool,
    include_ref: bool = True,
) -> None:
    """
    Single-color (blue-ish) trajectories for all actors in a run.
    If include_ref: overlay reference with dashed line (same color, lighter).
    """
    # Blue-ish as requested
    run_color = (0.10, 0.55, 0.90, 0.85)   # water/sky blue
    ref_color = (0.10, 0.55, 0.90, 0.35)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot run
    for aid, g in run_norm.sort_values(["actor_id", "frame_key"]).groupby("actor_id"):
        ax.plot(g["x_run"], g["y_run"], color=run_color, linewidth=1.6 if paper else 1.2)

    # Plot ref
    if include_ref:
        for aid, g in ref_norm.sort_values(["actor_id", "frame_key"]).groupby("actor_id"):
            ax.plot(g["x_ref"], g["y_ref"], color=ref_color, linewidth=1.2 if paper else 1.0, linestyle="--")

    label_fs = 22 if paper else 14
    tick_fs = 20 if paper else 12

    ax.set_xlabel("X [m]", fontsize=label_fs)
    ax.set_ylabel("Y [m]", fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)
    ax.tick_params(direction="in")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.35)

    if not paper:
        ax.set_title(tag)

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, required=True, help="Sweep results directory (e.g., sweep_results_YYYY...).")
    p.add_argument("--ref", type=Path, required=True, help="Reference CSV (sender input, e.g., exp_300.csv).")
    p.add_argument("--paper", action="store_true", help="Paper styling (fonts, sizes). Output is PDF.")
    p.add_argument("--per-actor", action="store_true", help="Also output per-actor RMSE CSV per run.")
    p.add_argument(
        "--no-ref-overlay",
        action="store_true",
        help="Do not overlay reference trajectories (still computes RMSE).",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    outdir = args.outdir
    ref_path = args.ref

    if args.paper:
        setup_paper_fonts()

    if not outdir.exists():
        raise SystemExit(f"[ERROR] outdir not found: {outdir}")
    if not ref_path.exists():
        raise SystemExit(f"[ERROR] ref not found: {ref_path}")

    runs = discover_runs(outdir)
    if not runs:
        raise SystemExit(f"[ERROR] No run CSVs found under: {outdir}")

    # Normalize reference once
    ref_norm = normalize_ref(ref_path)

    # Timing fallback: receiver_stderr at top-level (if exists)
    receiver_stderr = outdir / "receiver_stderr.log"

    summary_rows = []

    for r in runs:
        n, ts = parse_tag(r.tag)

        run_norm = normalize_run(r.state_csv)

        rmse_xy, rmse_x, rmse_y, n_match = compute_rmse(ref_norm, run_norm)

        # timing: prefer per-run stream_timing
        mean_ms = p50_ms = p95_ms = None
        n_timing = 0
        if r.timing_csv and r.timing_csv.exists():
            mean_ms, p50_ms, p95_ms, n_timing = timing_from_stream_csv(r.timing_csv)

        if (mean_ms is None) and receiver_stderr.exists():
            # Fallback to receiver log (not per-run, but better than nothing)
            mean_ms, p50_ms, p95_ms, n_timing = timing_from_receiver_stderr(receiver_stderr)

        # Output trajectory PDF
        out_pdf = r.run_dir / f"trajectories_{r.tag}.pdf"
        plot_run_trajectories_pdf(
            out_pdf=out_pdf,
            tag=r.tag,
            ref_norm=ref_norm,
            run_norm=run_norm,
            paper=args.paper,
            include_ref=(not args.no_ref_overlay),
        )

        # Optional per-actor rmse
        if args.per_actor:
            per = compute_rmse_per_actor(ref_norm, run_norm)
            per_path = r.run_dir / f"per_actor_rmse_{r.tag}.csv"
            per.to_csv(per_path, index=False, encoding="utf-8-sig")

        summary_rows.append(
            {
                "tag": r.tag,
                "N": n if n is not None else "",
                "Ts": ts if ts is not None else "",
                "state_csv": str(r.state_csv),
                "timing_csv": str(r.timing_csv) if r.timing_csv else "",
                "rmse_xy": rmse_xy,
                "rmse_x": rmse_x,
                "rmse_y": rmse_y,
                "n_matched": n_match,
                "frame_ms_mean": mean_ms if mean_ms is not None else "",
                "frame_ms_p50": p50_ms if p50_ms is not None else "",
                "frame_ms_p95": p95_ms if p95_ms is not None else "",
                "n_timing": n_timing,
                "traj_pdf": str(out_pdf),
            }
        )

    summary = pd.DataFrame(summary_rows)

    # Sort nicely if N/Ts available
    def _sort_key(v):
        try:
            return float(v)
        except Exception:
            return float("inf")

    if "N" in summary.columns and "Ts" in summary.columns:
        # Convert for sort (keep blank as inf)
        summary["_N_sort"] = summary["N"].apply(_sort_key)
        summary["_Ts_sort"] = summary["Ts"].apply(_sort_key)
        summary = summary.sort_values(["_N_sort", "_Ts_sort", "tag"]).drop(columns=["_N_sort", "_Ts_sort"])

    out_summary = outdir / "summary_runs.csv"
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")

    print(f"[OK] Wrote: {out_summary}")
    print(f"[OK] Runs: {len(runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
