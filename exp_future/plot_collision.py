#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """
    Otsu threshold on positive scalar values (expects already preprocessed, e.g., log1p).
    Returns threshold in the same space as `values`.
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan")

    hist, bin_edges = np.histogram(v, bins=bins)
    hist = hist.astype(np.float64)
    p = hist / max(hist.sum(), 1.0)

    omega = np.cumsum(p)
    mu = np.cumsum(p * (bin_edges[:-1] + bin_edges[1:]) * 0.5)
    mu_t = mu[-1]

    sigma_b2 = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
    k = int(np.nanargmax(sigma_b2))
    thr = (bin_edges[k] + bin_edges[k + 1]) * 0.5
    return float(thr)


def robust_threshold_iqr(x: np.ndarray, k: float = 3.0) -> float:
    """Q3 + k*IQR as a robust outlier-ish threshold."""
    v = x[np.isfinite(x)]
    if v.size == 0:
        return float("nan")
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    return float(q3 + k * iqr)


def canonical_other_class(other_type: str) -> str:
    s = (other_type or "").lower()
    if s.startswith("vehicle."):
        return "vehicle"
    if s.startswith("walker."):
        return "walker"
    if s.startswith("traffic."):
        return "traffic"
    if s.startswith("static."):
        return "static"
    if s.strip() == "":
        return "unknown"
    return "other"


def summarize_group(df: pd.DataFrame, name: str) -> Dict[str, float]:
    x = df["intensity"].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0}
    pct = np.percentile(x, [50, 75, 90, 95, 99])
    return {
        "count": float(x.size),
        "min": float(x.min()),
        "p50": float(pct[0]),
        "p75": float(pct[1]),
        "p90": float(pct[2]),
        "p95": float(pct[3]),
        "p99": float(pct[4]),
        "max": float(x.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("collisions.csv"))
    ap.add_argument("--bins", type=int, default=120)
    ap.add_argument("--min-intensity", type=float, default=0.0, help="Filter out intensities below this.")
    ap.add_argument("--show-plots", action="store_true", help="Show plots interactively.")
    args = ap.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"not found: {args.csv}")

    df = pd.read_csv(args.csv)

    required = {"time_sec", "frame", "vehicle_id", "other_type", "x", "y", "z", "intensity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
    df = df[np.isfinite(df["intensity"])]
    df = df[df["intensity"] >= float(args.min_intensity)].copy()

    df["other_class"] = df["other_type"].astype(str).map(canonical_other_class)

    if len(df) == 0:
        print("No collision rows after filtering.")
        return

    # ---- summary (overall) ----
    overall = summarize_group(df, "overall")
    print("=== Overall intensity summary (N*s) ===")
    for k in ["count", "min", "p50", "p75", "p90", "p95", "p99", "max"]:
        if k in overall:
            print(f"{k:>6}: {overall[k]:.3f}" if k != "count" else f"{k:>6}: {int(overall[k])}")

    # Percentile-based threshold candidates
    x = df["intensity"].to_numpy(dtype=float)
    p90, p95, p99 = np.percentile(x, [90, 95, 99])

    # Robust IQR threshold
    thr_iqr = robust_threshold_iqr(x, k=3.0)

    # Otsu on log1p(intensity) (helps separate light contacts vs strong impacts)
    lx = np.log1p(x)
    thr_l_otsu = otsu_threshold(lx, bins=256)
    thr_otsu = float(np.expm1(thr_l_otsu)) if np.isfinite(thr_l_otsu) else float("nan")

    print("\n=== Threshold candidates (N*s) ===")
    print(f"p90 : {p90:.3f}")
    print(f"p95 : {p95:.3f}")
    print(f"p99 : {p99:.3f}")
    print(f"IQR : {thr_iqr:.3f}  (Q3 + 3*IQR)")
    print(f"Otsu: {thr_otsu:.3f}  (on log1p(intensity))")

    # ---- per other_class ----
    print("\n=== By other_class intensity summary (N*s) ===")
    grp = df.groupby("other_class", dropna=False)
    rows = []
    for name, g in grp:
        s = summarize_group(g, str(name))
        s["other_class"] = name
        rows.append(s)
    s_df = pd.DataFrame(rows).fillna(0.0).sort_values("count", ascending=False)

    # print compact table
    cols = ["other_class", "count", "p50", "p90", "p95", "p99", "max"]
    print(s_df[cols].to_string(index=False, formatters={
        "count": lambda v: f"{int(v)}",
        "p50": lambda v: f"{v:.3f}",
        "p90": lambda v: f"{v:.3f}",
        "p95": lambda v: f"{v:.3f}",
        "p99": lambda v: f"{v:.3f}",
        "max": lambda v: f"{v:.3f}",
    }))

    # ---- plots ----
    out_png1 = args.csv.with_suffix(".intensity_hist.png")
    out_png2 = args.csv.with_suffix(".intensity_log_hist.png")

    plt.figure()
    plt.hist(x, bins=args.bins)
    plt.xlabel("Intensity |normal_impulse| [N*s]")
    plt.ylabel("Count")
    plt.title("Collision intensity histogram")
    # draw candidate thresholds
    for t, label in [(p95, "p95"), (thr_iqr, "IQR"), (thr_otsu, "Otsu")]:
        if np.isfinite(t):
            plt.axvline(t, linestyle="--", label=label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png1, dpi=200)

    plt.figure()
    plt.hist(np.log1p(x), bins=args.bins)
    plt.xlabel("log(1 + intensity)")
    plt.ylabel("Count")
    plt.title("Collision intensity histogram (log scale)")
    if np.isfinite(thr_l_otsu):
        plt.axvline(thr_l_otsu, linestyle="--", label="Otsu (log)")
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png2, dpi=200)

    print(f"\nSaved plots:")
    print(f"- {out_png1}")
    print(f"- {out_png2}")

    # ---- practical suggestion ----
    # pick a conservative default suggestion:
    # use Otsu if it exists; otherwise p95.
    suggested = thr_otsu if np.isfinite(thr_otsu) else float(p95)
    print("\n=== Suggested starting threshold ===")
    print(f"{suggested:.3f} N*s  (use this as 'accident' threshold, then validate with a few scenes)")

    if args.show_plots:
        plt.show()


if __name__ == "__main__":
    main()
