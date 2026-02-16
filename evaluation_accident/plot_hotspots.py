#!/usr/bin/env python3
# Plot hotspot collision events on XY plane from nearby_hotspot_events_with_dist.csv
#
# Usage:
#   python plot_hotspots.py nearby_hotspot_events_with_dist.csv --out hotspot.png
#   python plot_hotspots.py nearby_hotspot_events_with_dist.csv --method lstm --lead 10
#
# Notes:
# - Assumes columns include at least: x, y, method, lead_sec (or lead), dist_m (distance to hotspot)
# - If your CSV uses different column names, adjust the CANDIDATES dict below.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


CANDIDATES: Dict[str, List[str]] = {
    "x": ["x", "location_x", "pos_x", "world_x"],
    "y": ["y", "location_y", "pos_y", "world_y"],
    "method": ["method", "future_mode", "mode"],
    "lead": ["lead_sec", "lead", "lead_seconds"],
    "dist": ["dist_m", "distance_m", "distance", "dist"],
    "intensity": ["intensity", "collision_intensity"],
    "radius": ["radius_m", "radius", "r_m"],
}


def pick_col(df: pd.DataFrame, key: str) -> Optional[str]:
    for c in CANDIDATES.get(key, []):
        if c in df.columns:
            return c
    return None


def require_cols(df: pd.DataFrame, keys: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    missing: List[str] = []
    for k in keys:
        col = pick_col(df, k)
        if col is None:
            missing.append(k)
        else:
            mapping[k] = col
    if missing:
        raise SystemExit(
            f"Missing required columns for: {missing}\n"
            f"Available columns:\n  {', '.join(df.columns.astype(str))}\n"
            f"Edit CANDIDATES in this script to match your CSV."
        )
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=str, help="nearby hotspot events CSV (e.g., nearby_hotspot_events_with_dist.csv)")
    ap.add_argument("--out", type=str, default="hotspot.png", help="output image path")
    ap.add_argument("--method", type=str, default="", help="filter method (autopilot/lstm). empty=both")
    ap.add_argument("--lead", type=int, default=-1, help="filter lead_sec. -1=all")
    ap.add_argument("--radius", type=float, default=-1, help="filter by radius_m if present. -1=ignore")
    ap.add_argument("--max-points", type=int, default=200000, help="downsample cap for plotting")
    ap.add_argument("--alpha", type=float, default=0.15, help="point alpha")
    ap.add_argument("--s", type=float, default=3.0, help="point size")
    ap.add_argument("--title", type=str, default="", help="override plot title")
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cols = require_cols(df, ["x", "y", "method", "lead", "dist"])

    # Basic cleaning
    for k in ["x", "y", "dist", "lead"]:
        df[cols[k]] = pd.to_numeric(df[cols[k]], errors="coerce")
    df = df.dropna(subset=[cols["x"], cols["y"], cols["dist"], cols["lead"]])

    # Filters
    if args.method:
        df = df[df[cols["method"]].astype(str).str.lower() == args.method.lower()]
    if args.lead >= 0:
        df = df[df[cols["lead"]].astype(int) == int(args.lead)]

    r_col = pick_col(df, "radius")
    if args.radius >= 0:
        if r_col is None:
            raise SystemExit("--radius was given but radius column not found in CSV.")
        df[r_col] = pd.to_numeric(df[r_col], errors="coerce")
        df = df[df[r_col] == float(args.radius)]

    # Downsample if huge
    n = len(df)
    if n == 0:
        raise SystemExit("No rows to plot after filters.")
    if n > args.max_points:
        df = df.sample(args.max_points, random_state=0)

    # Plot
    fig = plt.figure(figsize=(9, 7))
    ax = plt.gca()

    # Color by method if both present; otherwise single series
    methods = sorted(df[cols["method"]].astype(str).str.lower().unique().tolist())
    if len(methods) <= 1:
        ax.scatter(df[cols["x"]], df[cols["y"]], s=args.s, alpha=args.alpha)
    else:
        for m in methods:
            sub = df[df[cols["method"]].astype(str).str.lower() == m]
            ax.scatter(sub[cols["x"]], sub[cols["y"]], s=args.s, alpha=args.alpha, label=m)
        ax.legend(loc="best")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (world)")
    ax.set_ylabel("y (world)")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    title = args.title
    if not title:
        parts: List[str] = ["Hotspot-near collision events (XY)"]
        if args.method:
            parts.append(f"method={args.method}")
        if args.lead >= 0:
            parts.append(f"lead={args.lead}s")
        if args.radius >= 0:
            parts.append(f"radius={args.radius}m")
        title = " | ".join(parts)
    ax.set_title(title)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
