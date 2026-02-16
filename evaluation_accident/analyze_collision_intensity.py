#!/usr/bin/env python3
"""
Analyze and plot CARLA collision intensity distributions from a CSV.

Features:
- Histogram + ECDF for intensity (overall)
- Overlay histogram by group (e.g., method)
- Per-lead_sec plots (one file per lead_sec), optional overlay by method
- Threshold sweep counts:
    * overall
    * by method
    * by method x lead_sec
- Optional event-level de-duplication:
    Many collision logs contain two rows for the same collision:
      (actor_id=A, other_id=B) and (actor_id=B, other_id=A)
    This script can collapse them into 1 "event" using (carla_frame, method, lead_sec, rep, sorted_pair_id).

Assumptions:
- Input CSV contains an intensity column (default: 'intensity').
- If no intensity column exists, the script attempts to compute it from impulse components.

Outputs:
- PNGs for histograms and ECDF
- CSV summaries for stats and threshold sweep
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -------------------------
# Column detection helpers
# -------------------------

def find_intensity_column(df: pd.DataFrame) -> Optional[str]:
    for name in ("intensity", "collision_intensity", "impact_intensity"):
        if name in df.columns:
            return name
    candidates = [c for c in df.columns if "intensity" in c.lower()]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return sorted(candidates, key=len)[0]
    return None


def find_impulse_columns(df: pd.DataFrame) -> Optional[Tuple[str, str, str]]:
    lower = {c.lower(): c for c in df.columns}
    patterns = [
        ("normal_impulse_x", "normal_impulse_y", "normal_impulse_z"),
        ("normal_impulse.x", "normal_impulse.y", "normal_impulse.z"),
        ("impulse_x", "impulse_y", "impulse_z"),
    ]
    for xk, yk, zk in patterns:
        if xk in lower and yk in lower and zk in lower:
            return (lower[xk], lower[yk], lower[zk])

    xs = [c for c in df.columns if c.lower().endswith(("_x", ".x")) and "impulse" in c.lower()]
    ys = [c for c in df.columns if c.lower().endswith(("_y", ".y")) and "impulse" in c.lower()]
    zs = [c for c in df.columns if c.lower().endswith(("_z", ".z")) and "impulse" in c.lower()]

    def base(col: str) -> str:
        cl = col.lower()
        if cl.endswith("_x") or cl.endswith("_y") or cl.endswith("_z"):
            return cl[:-2]
        if cl.endswith(".x") or cl.endswith(".y") or cl.endswith(".z"):
            return cl[:-2]
        return cl

    if xs and ys and zs:
        for x in xs:
            bx = base(x)
            y = next((yy for yy in ys if base(yy) == bx), None)
            z = next((zz for zz in zs if base(zz) == bx), None)
            if y and z:
                return (x, y, z)
    return None


def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def compute_intensity_from_impulse(df: pd.DataFrame, cols: Tuple[str, str, str]) -> pd.Series:
    x, y, z = cols
    vx = coerce_numeric(df[x])
    vy = coerce_numeric(df[y])
    vz = coerce_numeric(df[z])
    return (vx * vx + vy * vy + vz * vz).pow(0.5)


# -------------------------
# Stats + plots
# -------------------------

def summary_stats(values: pd.Series) -> Dict[str, float]:
    v = values.dropna()
    if v.empty:
        return {"count": 0.0}
    q = v.quantile([0.5, 0.9, 0.95, 0.99])
    return {
        "count": float(v.shape[0]),
        "min": float(v.min()),
        "mean": float(v.mean()),
        "median": float(q.loc[0.5]),
        "p90": float(q.loc[0.9]),
        "p95": float(q.loc[0.95]),
        "p99": float(q.loc[0.99]),
        "max": float(v.max()),
    }


def save_hist(values: pd.Series, out_path: Path, bins: int, log_y: bool, title: str, x_max: Optional[float]) -> None:
    v = values.dropna()
    if x_max is not None:
        v = v[v <= x_max]
    plt.figure()
    plt.hist(v.to_numpy(), bins=bins)
    if log_y:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("intensity")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_ecdf(values: pd.Series, out_path: Path, title: str, x_max: Optional[float]) -> None:
    v = values.dropna().sort_values()
    if x_max is not None:
        v = v[v <= x_max]
    n = len(v)
    if n == 0:
        return
    y = np.arange(1, n + 1) / n
    plt.figure()
    plt.plot(v.to_numpy(), y)
    plt.title(title)
    plt.xlabel("intensity")
    plt.ylabel("ECDF")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_overlay_hist_by_group(
    df: pd.DataFrame,
    intensity_col: str,
    group_col: str,
    out_path: Path,
    bins: int,
    log_y: bool,
    title: str,
    x_max: Optional[float],
) -> None:
    if group_col not in df.columns:
        raise ValueError(f"group column '{group_col}' not found. columns={list(df.columns)}")

    plt.figure()
    for key, sub in df.groupby(group_col, dropna=False):
        v = coerce_numeric(sub[intensity_col]).dropna()
        if x_max is not None:
            v = v[v <= x_max]
        if v.empty:
            continue
        plt.hist(v.to_numpy(), bins=bins, alpha=0.5, label=str(key))
    if log_y:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("intensity")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------
# Event de-duplication
# -------------------------

def has_cols(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return all(c in df.columns for c in cols)


def add_sorted_pair_id(df: pd.DataFrame, a_col: str, b_col: str) -> pd.Series:
    a = coerce_numeric(df[a_col])
    b = coerce_numeric(df[b_col])
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    # Use a safe string id; numeric packing can overflow if ids are big.
    return lo.astype("Int64").astype(str) + "-" + hi.astype("Int64").astype(str)


def dedup_to_events(
    df: pd.DataFrame,
    intensity_col: str,
    default_keys: Sequence[str],
    actor_id_col: str = "actor_id",
    other_id_col: str = "other_id",
) -> pd.DataFrame:
    missing = [k for k in default_keys if k not in df.columns]
    if missing:
        raise ValueError(f"Cannot deduplicate: missing key columns: {missing}")

    if not has_cols(df, [actor_id_col, other_id_col]):
        raise ValueError(f"Cannot deduplicate: need '{actor_id_col}' and '{other_id_col}' columns.")

    df2 = df.copy()
    df2["_pair_id"] = add_sorted_pair_id(df2, actor_id_col, other_id_col)

    # Group as one event, keep max intensity (most severe contact) by default.
    group_cols = list(default_keys) + ["_pair_id"]
    agg = df2.groupby(group_cols, dropna=False)[intensity_col].max().reset_index()
    agg.rename(columns={intensity_col: intensity_col}, inplace=True)
    return agg


# -------------------------
# Threshold sweep
# -------------------------

def parse_thresholds(s: str) -> List[float]:
    # comma-separated floats
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("threshold list is empty")
    return sorted(out)


def build_default_thresholds() -> List[float]:
    # Reasonable defaults for CARLA collision intensity experiments
    return [500, 1000, 1500, 2000, 3000, 4000, 5000, 7000, 10000, 15000, 20000]


def threshold_sweep_counts(
    df: pd.DataFrame,
    intensity_col: str,
    thresholds: Sequence[float],
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    v = df[intensity_col]
    rows: List[Dict[str, object]] = []
    if group_cols is None:
        group_cols = []

    if group_cols:
        grouped = df.groupby(list(group_cols), dropna=False)
        for gkey, sub in grouped:
            if not isinstance(gkey, tuple):
                gkey = (gkey,)
            gmap = {col: val for col, val in zip(group_cols, gkey)}
            sv = sub[intensity_col]
            for thr in thresholds:
                rows.append({**gmap, "threshold": float(thr), "count": int((sv >= thr).sum())})
    else:
        for thr in thresholds:
            rows.append({"threshold": float(thr), "count": int((v >= thr).sum())})

    return pd.DataFrame(rows)


def save_threshold_plot(
    df_counts: pd.DataFrame,
    out_path: Path,
    title: str,
    x_label: str = "threshold",
    y_label: str = "count",
    log_x: bool = False,
    log_y: bool = False,
    series_col: Optional[str] = None,
) -> None:
    plt.figure()
    if series_col and series_col in df_counts.columns:
        for key, sub in df_counts.groupby(series_col, dropna=False):
            sub2 = sub.sort_values("threshold")
            plt.plot(sub2["threshold"].to_numpy(), sub2["count"].to_numpy(), label=str(key))
        plt.legend()
    else:
        sub2 = df_counts.sort_values("threshold")
        plt.plot(sub2["threshold"].to_numpy(), sub2["count"].to_numpy())

    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------
# CLI
# -------------------------

@dataclass
class Config:
    csv: Path
    outdir: Path
    prefix: str
    bins: int
    log_y: bool
    x_max: Optional[float]
    min_intensity: Optional[float]
    group_by: Optional[str]
    facet_by: Optional[str]
    facet_overlay_by: Optional[str]
    thresholds: List[float]
    dedup_events: bool
    event_keys: List[str]
    plot_thresholds: bool
    log_x_threshold_plot: bool
    log_y_threshold_plot: bool


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument("--outdir", default="out", help="Output directory")
    p.add_argument("--prefix", default="intensity", help="Output file prefix")
    p.add_argument("--bins", type=int, default=120, help="Histogram bins")
    p.add_argument("--log-y", action="store_true", help="Histogram y-axis log scale")
    p.add_argument("--x-max", type=float, default=None, help="Clip plotted intensities at this max")
    p.add_argument("--min-intensity", type=float, default=None, help="Filter: keep rows with intensity >= this")

    p.add_argument("--group-by", default="method", help="Overlay histogram group column (default: method)")
    p.add_argument("--facet-by", default="lead_sec", help="Facet plots by this column (default: lead_sec). Use 'none' to disable.")
    p.add_argument("--facet-overlay-by", default="method", help="Within each facet, overlay by this column (default: method). Use 'none' to disable.")

    p.add_argument(
        "--thresholds",
        default="",
        help="Comma-separated thresholds, e.g. '1000,2000,5000,10000'. Empty = use defaults.",
    )
    p.add_argument("--plot-thresholds", action="store_true", help="Also output threshold-sweep line plots")
    p.add_argument("--log-x-threshold-plot", action="store_true", help="Log scale for threshold axis")
    p.add_argument("--log-y-threshold-plot", action="store_true", help="Log scale for count axis")

    p.add_argument("--dedup-events", action="store_true", help="Collapse symmetric (actor,other) pairs into 1 event")
    p.add_argument(
        "--event-keys",
        default="method,lead_sec,rep,carla_frame",
        help="Keys for event grouping when --dedup-events is set (default: method,lead_sec,rep,carla_frame)",
    )

    a = p.parse_args(argv)

    thresholds = parse_thresholds(a.thresholds) if a.thresholds.strip() else build_default_thresholds()

    facet_by = None if str(a.facet_by).lower() == "none" else str(a.facet_by)
    facet_overlay_by = None if str(a.facet_overlay_by).lower() == "none" else str(a.facet_overlay_by)
    group_by = None if str(a.group_by).lower() == "none" else str(a.group_by)

    return Config(
        csv=Path(a.csv),
        outdir=Path(a.outdir),
        prefix=str(a.prefix),
        bins=int(a.bins),
        log_y=bool(a.log_y),
        x_max=a.x_max,
        min_intensity=a.min_intensity,
        group_by=group_by,
        facet_by=facet_by,
        facet_overlay_by=facet_overlay_by,
        thresholds=thresholds,
        dedup_events=bool(a.dedup_events),
        event_keys=[x.strip() for x in str(a.event_keys).split(",") if x.strip()],
        plot_thresholds=bool(a.plot_thresholds),
        log_x_threshold_plot=bool(a.log_x_threshold_plot),
        log_y_threshold_plot=bool(a.log_y_threshold_plot),
    )


def main() -> int:
    cfg = parse_args()
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.csv)

    intensity_col = find_intensity_column(df)
    if intensity_col is None:
        impulse_cols = find_impulse_columns(df)
        if impulse_cols is None:
            raise ValueError(
                "No intensity column found, and could not find impulse component columns to compute it.\n"
                f"CSV columns: {list(df.columns)}"
            )
        df["intensity"] = compute_intensity_from_impulse(df, impulse_cols)
        intensity_col = "intensity"

    df[intensity_col] = coerce_numeric(df[intensity_col])

    # Filter (row-level)
    if cfg.min_intensity is not None:
        df = df[df[intensity_col] >= float(cfg.min_intensity)]

    # Row-level summary
    stats_overall = summary_stats(df[intensity_col])
    print(f"[summary rows] intensity_col={intensity_col}")
    for k in ("count", "min", "mean", "median", "p90", "p95", "p99", "max"):
        if k in stats_overall:
            print(f"{k:>6}: {stats_overall[k]}")

    # Save row-level stats by (method, lead_sec)
    stats_rows = []
    group_cols = [c for c in ["method", "lead_sec"] if c in df.columns]
    if group_cols:
        for gkey, sub in df.groupby(group_cols, dropna=False):
            if not isinstance(gkey, tuple):
                gkey = (gkey,)
            s = summary_stats(sub[intensity_col])
            row = {col: val for col, val in zip(group_cols, gkey)}
            row.update(s)
            stats_rows.append(row)
    if stats_rows:
        stats_df = pd.DataFrame(stats_rows).sort_values(group_cols)
        stats_path = cfg.outdir / f"{cfg.prefix}_stats_by_{'_'.join(group_cols)}.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"[saved] {stats_path}")

    # Overall plots (rows)
    save_hist(
        df[intensity_col],
        cfg.outdir / f"{cfg.prefix}_rows_hist.png",
        bins=cfg.bins,
        log_y=cfg.log_y,
        title="Intensity histogram (rows)",
        x_max=cfg.x_max,
    )
    save_ecdf(
        df[intensity_col],
        cfg.outdir / f"{cfg.prefix}_rows_ecdf.png",
        title="Intensity ECDF (rows)",
        x_max=cfg.x_max,
    )
    print(f"[saved] {cfg.outdir / f'{cfg.prefix}_rows_hist.png'}")
    print(f"[saved] {cfg.outdir / f'{cfg.prefix}_rows_ecdf.png'}")

    if cfg.group_by and cfg.group_by in df.columns:
        save_overlay_hist_by_group(
            df=df,
            intensity_col=intensity_col,
            group_col=cfg.group_by,
            out_path=cfg.outdir / f"{cfg.prefix}_rows_hist_by_{cfg.group_by}.png",
            bins=cfg.bins,
            log_y=cfg.log_y,
            title=f"Intensity histogram (rows) by {cfg.group_by}",
            x_max=cfg.x_max,
        )
        print(f"[saved] {cfg.outdir / f'{cfg.prefix}_rows_hist_by_{cfg.group_by}.png'}")

    # Facet plots (rows): one file per facet value
    if cfg.facet_by and cfg.facet_by in df.columns:
        facet_dir = cfg.outdir / f"{cfg.prefix}_facet_{cfg.facet_by}"
        facet_dir.mkdir(parents=True, exist_ok=True)
        for fval, sub in df.groupby(cfg.facet_by, dropna=False):
            safe = str(fval).replace("/", "_")
            # Histogram
            save_hist(
                sub[intensity_col],
                facet_dir / f"{cfg.prefix}_rows_{cfg.facet_by}_{safe}_hist.png",
                bins=cfg.bins,
                log_y=cfg.log_y,
                title=f"Intensity histogram (rows): {cfg.facet_by}={fval}",
                x_max=cfg.x_max,
            )
            # ECDF
            save_ecdf(
                sub[intensity_col],
                facet_dir / f"{cfg.prefix}_rows_{cfg.facet_by}_{safe}_ecdf.png",
                title=f"Intensity ECDF (rows): {cfg.facet_by}={fval}",
                x_max=cfg.x_max,
            )
            # Overlay within facet
            if cfg.facet_overlay_by and cfg.facet_overlay_by in sub.columns:
                save_overlay_hist_by_group(
                    df=sub,
                    intensity_col=intensity_col,
                    group_col=cfg.facet_overlay_by,
                    out_path=facet_dir / f"{cfg.prefix}_rows_{cfg.facet_by}_{safe}_hist_by_{cfg.facet_overlay_by}.png",
                    bins=cfg.bins,
                    log_y=cfg.log_y,
                    title=f"Rows hist: {cfg.facet_by}={fval} by {cfg.facet_overlay_by}",
                    x_max=cfg.x_max,
                )
        print(f"[saved] facet plots under: {facet_dir}")

    # Threshold sweep (rows)
    thr_overall = threshold_sweep_counts(df, intensity_col, cfg.thresholds)
    thr_overall_path = cfg.outdir / f"{cfg.prefix}_threshold_rows_overall.csv"
    thr_overall.to_csv(thr_overall_path, index=False)
    print(f"[saved] {thr_overall_path}")

    thr_method = None
    if "method" in df.columns:
        thr_method = threshold_sweep_counts(df, intensity_col, cfg.thresholds, group_cols=["method"])
        thr_method_path = cfg.outdir / f"{cfg.prefix}_threshold_rows_by_method.csv"
        thr_method.to_csv(thr_method_path, index=False)
        print(f"[saved] {thr_method_path}")

    thr_method_lead = None
    if has_cols(df, ["method", "lead_sec"]):
        thr_method_lead = threshold_sweep_counts(df, intensity_col, cfg.thresholds, group_cols=["method", "lead_sec"])
        thr_method_lead_path = cfg.outdir / f"{cfg.prefix}_threshold_rows_by_method_lead_sec.csv"
        thr_method_lead.to_csv(thr_method_lead_path, index=False)
        print(f"[saved] {thr_method_lead_path}")

    if cfg.plot_thresholds:
        save_threshold_plot(
            thr_overall,
            cfg.outdir / f"{cfg.prefix}_threshold_rows_overall.png",
            title="Threshold sweep (rows) - overall",
            log_x=cfg.log_x_threshold_plot,
            log_y=cfg.log_y_threshold_plot,
        )
        print(f"[saved] {cfg.outdir / f'{cfg.prefix}_threshold_rows_overall.png'}")

        if thr_method is not None:
            save_threshold_plot(
                thr_method,
                cfg.outdir / f"{cfg.prefix}_threshold_rows_by_method.png",
                title="Threshold sweep (rows) - by method",
                series_col="method",
                log_x=cfg.log_x_threshold_plot,
                log_y=cfg.log_y_threshold_plot,
            )
            print(f"[saved] {cfg.outdir / f'{cfg.prefix}_threshold_rows_by_method.png'}")

    # Event-level (optional)
    if cfg.dedup_events:
        events = dedup_to_events(df, intensity_col=intensity_col, default_keys=cfg.event_keys)
        print(f"[events] rows={len(df)} -> events={len(events)} (dedup enabled)")

        # Event-level plots
        save_hist(
            events[intensity_col],
            cfg.outdir / f"{cfg.prefix}_events_hist.png",
            bins=cfg.bins,
            log_y=cfg.log_y,
            title="Intensity histogram (events, deduplicated)",
            x_max=cfg.x_max,
        )
        save_ecdf(
            events[intensity_col],
            cfg.outdir / f"{cfg.prefix}_events_ecdf.png",
            title="Intensity ECDF (events, deduplicated)",
            x_max=cfg.x_max,
        )
        print(f"[saved] {cfg.outdir / f'{cfg.prefix}_events_hist.png'}")
        print(f"[saved] {cfg.outdir / f'{cfg.prefix}_events_ecdf.png'}")

        # Threshold sweep (events)
        thr_e_overall = threshold_sweep_counts(events, intensity_col, cfg.thresholds)
        thr_e_overall_path = cfg.outdir / f"{cfg.prefix}_threshold_events_overall.csv"
        thr_e_overall.to_csv(thr_e_overall_path, index=False)
        print(f"[saved] {thr_e_overall_path}")

        if "method" in events.columns:
            thr_e_method = threshold_sweep_counts(events, intensity_col, cfg.thresholds, group_cols=["method"])
            thr_e_method_path = cfg.outdir / f"{cfg.prefix}_threshold_events_by_method.csv"
            thr_e_method.to_csv(thr_e_method_path, index=False)
            print(f"[saved] {thr_e_method_path}")

            if cfg.plot_thresholds:
                save_threshold_plot(
                    thr_e_method,
                    cfg.outdir / f"{cfg.prefix}_threshold_events_by_method.png",
                    title="Threshold sweep (events) - by method",
                    series_col="method",
                    log_x=cfg.log_x_threshold_plot,
                    log_y=cfg.log_y_threshold_plot,
                )
                print(f"[saved] {cfg.outdir / f'{cfg.prefix}_threshold_events_by_method.png'}")

        if has_cols(events, ["method", "lead_sec"]):
            thr_e_ml = threshold_sweep_counts(events, intensity_col, cfg.thresholds, group_cols=["method", "lead_sec"])
            thr_e_ml_path = cfg.outdir / f"{cfg.prefix}_threshold_events_by_method_lead_sec.csv"
            thr_e_ml.to_csv(thr_e_ml_path, index=False)
            print(f"[saved] {thr_e_ml_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
