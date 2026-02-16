#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LEAD_RE = re.compile(r"lead_(\d+)$")
REP_RE = re.compile(r"rep_(\d+)$")

@dataclass
class RunRef:
    method: str
    lead: int
    rep: int
    actor_csv: Path
    collisions_csv: Path

def discover_runs(root: Path, methods: Optional[List[str]] = None) -> List[RunRef]:
    runs: List[RunRef] = []
    if methods is None:
        methods = ["autopilot", "lstm"]

    for method in methods:
        mdir = root / method
        if not mdir.exists():
            continue

        for lead_dir in sorted(mdir.glob("lead_*")):
            m = LEAD_RE.search(lead_dir.name)
            if not m:
                continue
            lead = int(m.group(1))

            for rep_dir in sorted(lead_dir.glob("rep_*")):
                r = REP_RE.search(rep_dir.name)
                if not r:
                    continue
                rep = int(r.group(1))

                logs = rep_dir / "logs"
                actor_csv = logs / "actor.csv"
                collisions_csv = logs / "collisions.csv"
                if actor_csv.exists() and collisions_csv.exists():
                    runs.append(RunRef(method, lead, rep, actor_csv, collisions_csv))
    return runs

# -------------------------------------------------------
# actor.csv -> near-miss PAIR count
# -------------------------------------------------------
def load_actor_csv_minimal(actor_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(actor_csv)

    required_any_id = ("carla_actor_id" in df.columns) or ("id" in df.columns)
    required = ["frame", "location_x", "location_y", "type"]
    missing = [c for c in required if c not in df.columns]
    if missing or not required_any_id:
        raise SystemExit(
            f"actor.csv schema mismatch: {actor_csv}\n"
            f"missing={missing}, has_id={required_any_id}\n"
            f"columns:\n  {', '.join(df.columns.astype(str))}"
        )

    df = df.copy()
    df["payload_frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["x"] = pd.to_numeric(df["location_x"], errors="coerce")
    df["y"] = pd.to_numeric(df["location_y"], errors="coerce")
    df["actor_type"] = df["type"].astype(str)

    if "carla_actor_id" in df.columns:
        df["actor_id"] = pd.to_numeric(df["carla_actor_id"], errors="coerce")
    else:
        df["actor_id"] = pd.to_numeric(df["id"], errors="coerce")

    df = df.dropna(subset=["payload_frame", "x", "y", "actor_id"]).copy()
    df["payload_frame"] = df["payload_frame"].astype(int)
    df["actor_id"] = df["actor_id"].astype(int)
    return df[["payload_frame", "x", "y", "actor_id", "actor_type"]]

def near_miss_pair_count_in_frame(
    xs: np.ndarray, ys: np.ndarray, ids: np.ndarray, threshold_m: float
) -> int:
    """
    Count ALL unordered pairs (i<j) with distance <= threshold in one frame.
    Uses grid bucketing. Returns pair count.
    """
    n = xs.shape[0]
    if n < 2:
        return 0

    thr = float(threshold_m)
    if thr <= 0:
        return 0

    cell = thr
    inv = 1.0 / cell
    cx = np.floor(xs * inv).astype(np.int64)
    cy = np.floor(ys * inv).astype(np.int64)

    buckets: Dict[Tuple[int, int], List[int]] = {}
    for i in range(n):
        buckets.setdefault((int(cx[i]), int(cy[i])), []).append(i)

    # Only check a half-neighborhood to avoid double counting across cells
    # (0,0) handles within-cell with j>i
    # (1,0),(0,1),(1,1),(1,-1) cover the rest uniquely
    neighbor_offsets = [(0, 0), (1, 0), (0, 1), (1, 1), (1, -1)]
    thr2 = thr * thr

    cnt = 0
    for (kx, ky), idxs in buckets.items():
        for dx, dy in neighbor_offsets:
            nb = (kx + dx, ky + dy)
            if nb not in buckets:
                continue
            jdxs = buckets[nb]

            if dx == 0 and dy == 0:
                # within same cell
                L = idxs
                m = len(L)
                for a in range(m):
                    i = L[a]
                    for b in range(a + 1, m):
                        j = L[b]
                        dxv = xs[i] - xs[j]
                        dyv = ys[i] - ys[j]
                        if dxv * dxv + dyv * dyv <= thr2:
                            cnt += 1
            else:
                # across two different cells (unique direction only)
                for i in idxs:
                    for j in jdxs:
                        dxv = xs[i] - xs[j]
                        dyv = ys[i] - ys[j]
                        if dxv * dxv + dyv * dyv <= thr2:
                            cnt += 1

    return int(cnt)

def count_nearmiss_pairs_total(
    actor_df: pd.DataFrame,
    switch_pf: int,
    threshold_m: float,
    actor_type_prefix: str,
    keep_every_k_frames: int,
) -> int:
    """
    Sum of near-miss pair counts across frames after switch_pf.
    1フレーム内の複数ペアも全部数える．
    """
    df = actor_df
    if actor_type_prefix:
        df = df[df["actor_type"].str.lower().str.startswith(actor_type_prefix.lower())]
    df = df[df["payload_frame"] >= switch_pf]
    if df.empty:
        return 0

    if keep_every_k_frames > 1:
        df = df[(df["payload_frame"] - switch_pf) % keep_every_k_frames == 0]
        if df.empty:
            return 0

    total = 0
    for _, sub in df.groupby("payload_frame", sort=True):
        xs = sub["x"].to_numpy(dtype=np.float64, copy=False)
        ys = sub["y"].to_numpy(dtype=np.float64, copy=False)
        ids = sub["actor_id"].to_numpy(dtype=np.int64, copy=False)
        total += near_miss_pair_count_in_frame(xs, ys, ids, threshold_m=threshold_m)

    return int(total)

# -------------------------------------------------------
# collisions.csv -> collision count (dedup symmetric)
# -------------------------------------------------------
def load_collisions_csv_minimal(collisions_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(collisions_csv)
    required = ["payload_frame", "actor_id", "other_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            f"collisions.csv schema mismatch: {collisions_csv}\n"
            f"missing={missing}\n"
            f"columns:\n  {', '.join(df.columns.astype(str))}"
        )

    df = df.copy()
    df["payload_frame"] = pd.to_numeric(df["payload_frame"], errors="coerce")
    df["actor_id"] = pd.to_numeric(df["actor_id"], errors="coerce")
    df["other_id"] = pd.to_numeric(df["other_id"], errors="coerce")

    if "is_accident" in df.columns:
        df["is_accident"] = pd.to_numeric(df["is_accident"], errors="coerce")
    if "intensity" in df.columns:
        df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")

    df = df.dropna(subset=["payload_frame", "actor_id", "other_id"]).copy()
    df["payload_frame"] = df["payload_frame"].astype(int)
    df["actor_id"] = df["actor_id"].astype(int)
    df["other_id"] = df["other_id"].astype(int)
    return df

def canonical_collision_key(df: pd.DataFrame) -> pd.Series:
    a = df["actor_id"].astype("Int64")
    b = df["other_id"].astype("Int64")
    lo = a.where(a <= b, b)
    hi = b.where(a <= b, a)
    base = df["payload_frame"].astype("Int64").astype(str) + "|" + lo.astype("Int64").astype(str) + "|" + hi.astype("Int64").astype(str)
    return base

def count_collisions(
    coll_df: pd.DataFrame,
    switch_pf: int,
    only_accident: bool,
    min_intensity: Optional[float],
) -> int:
    df = coll_df
    df = df[df["payload_frame"] >= switch_pf]
    if df.empty:
        return 0

    if only_accident:
        if "is_accident" not in df.columns:
            raise SystemExit("--only-accident was set but is_accident not found in collisions.csv")
        df = df.copy()
        df["is_accident"] = df["is_accident"].fillna(0).astype(int)
        df = df[df["is_accident"] == 1]
        if df.empty:
            return 0

    if min_intensity is not None:
        if "intensity" not in df.columns:
            raise SystemExit("--min-intensity was set but intensity not found in collisions.csv")
        df = df[df["intensity"].fillna(-1) >= float(min_intensity)]
        if df.empty:
            return 0

    df = df.copy()
    df["__key"] = canonical_collision_key(df)
    df = df.drop_duplicates(subset=["__key"])
    return int(len(df))

# -------------------------------------------------------
# main
# -------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results_grid_accident")
    ap.add_argument("--out-csv", type=str, default="", help="未指定なら root/out_counts_mean_over_rep_pairs.csv")

    ap.add_argument("--methods", type=str, nargs="*", default=None)
    ap.add_argument("--lead", type=int, default=-1)
    ap.add_argument("--rep", type=int, default=-1)

    ap.add_argument("--base-payload-frame", type=int, default=25411)
    ap.add_argument("--dt", type=float, default=0.1)

    ap.add_argument("--threshold-m", type=float, default=5.0)
    ap.add_argument("--actor-type-prefix", type=str, default="vehicle")
    ap.add_argument("--keep-every-k-frames", type=int, default=1)

    ap.add_argument("--only-accident", action="store_true")
    ap.add_argument("--min-intensity", type=float, default=None)

    args = ap.parse_args()

    fps = 1.0 / args.dt
    fps_i = int(round(fps))
    if abs(fps - fps_i) > 1e-6:
        raise SystemExit(f"dt={args.dt} -> fps={fps} not integer. set dt like 0.1")

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    runs = discover_runs(root, methods=args.methods)
    if args.lead >= 0:
        runs = [r for r in runs if r.lead == args.lead]
    if args.rep >= 0:
        runs = [r for r in runs if r.rep == args.rep]
    if not runs:
        raise SystemExit("No runs found (need both actor.csv and collisions.csv).")

    rows: List[Dict[str, object]] = []
    for run in runs:
        switch_pf = int(args.base_payload_frame - run.lead * fps_i)

        actor_df = load_actor_csv_minimal(run.actor_csv)
        coll_df = load_collisions_csv_minimal(run.collisions_csv)

        nearmiss_pairs_total = count_nearmiss_pairs_total(
            actor_df=actor_df,
            switch_pf=switch_pf,
            threshold_m=args.threshold_m,
            actor_type_prefix=args.actor_type_prefix,
            keep_every_k_frames=max(1, int(args.keep_every_k_frames)),
        )
        collision_n = count_collisions(
            coll_df=coll_df,
            switch_pf=switch_pf,
            only_accident=bool(args.only_accident),
            min_intensity=args.min_intensity,
        )

        rows.append({
            "method": run.method,
            "lead_sec": int(run.lead),
            "rep": int(run.rep),
            "switch_payload_frame": int(switch_pf),
            "nearmiss_pairs_total": int(nearmiss_pairs_total),
            "collision_count": int(collision_n),
        })

        print(f"[OK] {run.method} lead={run.lead} rep={run.rep} nearmiss_pairs={nearmiss_pairs_total} collision={collision_n}")

    per_run = pd.DataFrame(rows).sort_values(["method", "lead_sec", "rep"]).reset_index(drop=True)

    # repごとに出した値の平均（method × lead）
    summary = (
        per_run
        .groupby(["method", "lead_sec"], as_index=False)
        .agg(
            reps=("rep", "nunique"),
            nearmiss_pairs_mean=("nearmiss_pairs_total", "mean"),
            nearmiss_pairs_std=("nearmiss_pairs_total", "std"),
            collision_mean=("collision_count", "mean"),
            collision_std=("collision_count", "std"),
        )
        .sort_values(["method", "lead_sec"])
        .reset_index(drop=True)
    )

    # 全leadまとめたmethod別平均（参考）
    summary_method = (
        per_run
        .groupby(["method"], as_index=False)
        .agg(
            runs=("rep", "count"),
            nearmiss_pairs_mean=("nearmiss_pairs_total", "mean"),
            nearmiss_pairs_std=("nearmiss_pairs_total", "std"),
            collision_mean=("collision_count", "mean"),
            collision_std=("collision_count", "std"),
        )
        .sort_values(["method"])
        .reset_index(drop=True)
    )

    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else (root / "out_counts_mean_over_rep_pairs.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    block1 = per_run.copy()
    block1.insert(0, "table", "per_run")

    block2 = summary.copy()
    block2.insert(0, "table", "mean_over_rep_by_method_lead")

    block3 = summary_method.copy()
    block3.insert(0, "table", "mean_over_rep_by_method_all_leads")

    out = pd.concat([block1, block2, block3], ignore_index=True, sort=False)
    out.to_csv(out_csv, index=False)
    print(f"[OK] Wrote: {out_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
