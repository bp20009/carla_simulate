#!/usr/bin/env python3
"""
Plot collision intensity distributions from a CSV.

- If an 'intensity' column exists, use it.
- Otherwise, try to compute intensity from normal impulse components (x/y/z).
Outputs:
  - Histogram PNG (overall)
  - Histogram by group (e.g., method) PNG (optional)
  - ECDF PNG (overall)
Also prints summary statistics to stdout.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _find_intensity_column(df: pd.DataFrame) -> Optional[str]:
    # Prefer exact matches first
    for name in ("intensity", "collision_intensity", "impact_intensity"):
        if name in df.columns:
            return name
    # Heuristic: any column containing "intensity"
    candidates = [c for c in df.columns if "intensity" in c.lower()]
    if len(candidates) == 1:
        return candidates[0]
    # If multiple, pick the shortest (often the canonical one)
    if candidates:
        return sorted(candidates, key=len)[0]
    return None


def _find_impulse_columns(df: pd.DataFrame) -> Optional[Tuple[str, str, str]]:
    lower = {c.lower(): c for c in df.columns}

    # Common patterns:
    # - normal_impulse_x, normal_impulse_y, normal_impulse_z
    # - normalImpulseX (unlikely in CSV), etc.
    patterns = [
        ("normal_impulse_x", "normal_impulse_y", "normal_impulse_z"),
        ("normal_impulse.x", "normal_impulse.y", "normal_impulse.z"),
        ("impulse_x", "impulse_y", "impulse_z"),
    ]
    for xk, yk, zk in patterns:
        if xk in lower and yk in lower and zk in lower:
            return (lower[xk], lower[yk], lower[zk])

    # More flexible: search for columns that end with _x/_y/_z and contain "impulse"
    xs = [c for c in df.columns if c.lower().endswith(("_x", ".x")) and "impulse" in c.lower()]
    ys = [c for c in df.columns if c.lower().endswith(("_y", ".y")) and "impulse" in c.lower()]
    zs = [c for c in df.columns if c.lower().endswith(("_z", ".z")) and "impulse" in c.lower()]

    if xs and ys and zs:
        # Pick the most similar prefixes by stripping suffix
        def base(col: str) -> str:
            cl = col.lower()
            if cl.endswith("_x"):
                return cl[:-2]
            if cl.endswith(".x"):
                return cl[:-2]
            return cl

        for x in xs:
            bx = base(x)
            y_match = next((y for y in ys if base(y) == bx), None)
            z_match = next((z for z in zs if base(z) == bx), None)
            if y_match and z_match:
                return (x, y_match, z_match)

    return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _compute_intensity_from_impulse(df: pd.DataFrame, cols: Tuple[str, str, str]) -> pd.Series:
    x, y, z = cols
    vx = _coerce_numeric(df[x])
    vy = _coerce_numeric(df[y])
    vz = _coerce_numeric(df[z])
    return (vx * vx + vy * vy + vz * vz).pow(0.5)


def _summary_stats(values: pd.Series) -> dict:
    v = values.dropna()
    if v.empty:
        return {"count": 0}
    qs = v.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    return {
        "count": int(v.shape[0]),
        "min": float(v.min()),
        "mean": float(v.mean()),
        "median": float(qs.get(0.5, float("nan"))),
        "p90": float(qs.get(0.9, float("nan"))),
        "p95": float(qs.get(0.95, float("nan"))),
        "p99": float(qs.get(0.99, float("nan"))),
        "max": float(v.max()),
    }


def _plot_hist(
    values: pd.Series,
    out_path: Path,
    bins: int,
    log_y: bool,
    title: str,
    x_label: str = "intensity",
    x_max: Optional[float] = None,
) -> None:
    v = values.dropna()
    if x_max is not None:
        v = v[v <= x_max]

    plt.figure()
    plt.hist(v.to_numpy(), bins=bins)
    if log_y:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_ecdf(
    values: pd.Series,
    out_path: Path,
    title: str,
    x_label: str = "intensity",
    x_max: Optional[float] = None,
) -> None:
    v = values.dropna().sort_values()
    if x_max is not None:
        v = v[v <= x_max]
    n = len(v)
    if n == 0:
        return

    y = (pd.Series(range(1, n + 1), index=v.index) / n).to_numpy()
    plt.figure()
    plt.plot(v.to_numpy(), y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("ECDF")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_hist_by_group(
    df: pd.DataFrame,
    intensity_col: str,
    group_col: str,
    out_path: Path,
    bins: int,
    log_y: bool,
    x_max: Optional[float],
) -> None:
    if group_col not in df.columns:
        raise ValueError(f"group column '{group_col}' not found in CSV columns: {list(df.columns)}")

    plt.figure()
    for key, sub in df.groupby(group_col, dropna=False):
        v = pd.to_numeric(sub[intensity_col], errors="coerce").dropna()
        if x_max is not None:
            v = v[v <= x_max]
        if v.empty:
            continue
        # Overlay histograms
        plt.hist(v.to_numpy(), bins=bins, alpha=0.5, label=str(key))

    if log_y:
        plt.yscale("log")
    plt.title(f"Intensity histogram by {group_col}")
    plt.xlabel("intensity")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument("--outdir", default=".", help="Output directory")
    p.add_argument("--prefix", default="intensity", help="Output file prefix")
    p.add_argument("--bins", type=int, default=100, help="Histogram bins")
    p.add_argument("--log-y", action="store_true", help="Use log scale for Y axis (histogram)")
    p.add_argument("--x-max", type=float, default=None, help="Clip x-axis / values at this max for plots")
    p.add_argument("--min-intensity", type=float, default=None, help="Filter: keep rows with intensity >= this")
    p.add_argument("--group-by", default=None, help="Optional group column (e.g., method) for overlay histogram")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    intensity_col = _find_intensity_column(df)
    if intensity_col is None:
        impulse_cols = _find_impulse_columns(df)
        if impulse_cols is None:
            raise ValueError(
                "No intensity column found, and could not find impulse component columns to compute it.\n"
                f"CSV columns: {list(df.columns)}"
            )
        df["intensity"] = _compute_intensity_from_impulse(df, impulse_cols)
        intensity_col = "intensity"

    df[intensity_col] = pd.to_numeric(df[intensity_col], errors="coerce")

    if args.min_intensity is not None:
        df = df[df[intensity_col] >= float(args.min_intensity)]

    stats = _summary_stats(df[intensity_col])
    print(f"[summary] intensity column = {intensity_col}")
    for k in ("count", "min", "mean", "median", "p90", "p95", "p99", "max"):
        if k in stats:
            print(f"{k:>6}: {stats[k]}")

    # Outputs
    hist_path = outdir / f"{args.prefix}_hist.png"
    ecdf_path = outdir / f"{args.prefix}_ecdf.png"
    _plot_hist(
        df[intensity_col],
        out_path=hist_path,
        bins=args.bins,
        log_y=args.log_y,
        title="Intensity histogram",
        x_max=args.x_max,
    )
    _plot_ecdf(
        df[intensity_col],
        out_path=ecdf_path,
        title="Intensity ECDF",
        x_max=args.x_max,
    )

    if args.group_by:
        group_path = outdir / f"{args.prefix}_hist_by_{args.group_by}.png"
        _plot_hist_by_group(
            df=df,
            intensity_col=intensity_col,
            group_col=args.group_by,
            out_path=group_path,
            bins=args.bins,
            log_y=args.log_y,
            x_max=args.x_max,
        )

    print(f"[saved] {hist_path}")
    print(f"[saved] {ecdf_path}")
    if args.group_by:
        print(f"[saved] {outdir / f'{args.prefix}_hist_by_{args.group_by}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
