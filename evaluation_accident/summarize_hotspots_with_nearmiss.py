#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -------------------------------------------------------
# Discover: results_grid_accident/{method}/lead_*/rep_*/logs/{actor.csv, collisions.csv}
# -------------------------------------------------------
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
# actor.csv schema (your actual header)
#   frame, frame_source, carla_frame, id, carla_actor_id, type, location_x, location_y, ...
# We'll use:
#   payload_frame := frame
#   actor_id := carla_actor_id (fallback id)
#   actor_type := type
#   x,y := location_x, location_y
# -------------------------------------------------------
def load_actor_csv(actor_csv: Path) -> pd.DataFrame:
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

# -------------------------------------------------------
# collisions.csv schema (your earlier example)
#   time_sec,payload_frame,...,actor_id,actor_type,other_id,other_type,x,y,intensity,is_accident
# We'll require at least:
#   payload_frame,x,y,actor_id,other_id
# -------------------------------------------------------
def load_collisions_csv(collisions_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(collisions_csv)
    required = ["payload_frame", "x", "y", "actor_id", "other_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            f"collisions.csv schema mismatch: {collisions_csv}\n"
            f"missing={missing}\n"
            f"columns:\n  {', '.join(df.columns.astype(str))}"
        )

    df = df.copy()
    df["payload_frame"] = pd.to_numeric(df["payload_frame"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["actor_id"] = pd.to_numeric(df["actor_id"], errors="coerce")
    df["other_id"] = pd.to_numeric(df["other_id"], errors="coerce")

    if "intensity" in df.columns:
        df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
    if "is_accident" in df.columns:
        df["is_accident"] = pd.to_numeric(df["is_accident"], errors="coerce")

    df = df.dropna(subset=["payload_frame", "x", "y", "actor_id", "other_id"]).copy()
    df["payload_frame"] = df["payload_frame"].astype(int)
    df["actor_id"] = df["actor_id"].astype(int)
    df["other_id"] = df["other_id"].astype(int)
    return df

# -------------------------------------------------------
# Near-miss detection (grid neighbor search) - ALL PAIRS
# -------------------------------------------------------
def near_miss_pairs_midpoints(
    xs: np.ndarray, ys: np.ndarray, ids: np.ndarray, threshold_m: float
) -> List[Tuple[float, float, float, int, int]]:
    """
    1フレーム内の距離<=thresholdの全ペア（i<j）を列挙し，
    各ペアの中点(mx,my)と距離dist，(id_i,id_j)を返す．
    """
    n = xs.shape[0]
    if n < 2:
        return []

    thr = float(threshold_m)
    if thr <= 0:
        return []

    cell = thr
    inv = 1.0 / cell

    cx = np.floor(xs * inv).astype(np.int64)
    cy = np.floor(ys * inv).astype(np.int64)

    buckets: Dict[Tuple[int, int], List[int]] = {}
    for i in range(n):
        buckets.setdefault((int(cx[i]), int(cy[i])), []).append(i)

    thr2 = thr * thr

    # 重複回避のためセル間を片方向だけ調べる
    neighbor_offsets = [(0, 0), (1, 0), (0, 1), (1, 1), (1, -1)]

    out: List[Tuple[float, float, float, int, int]] = []

    for (kx, ky), idxs in buckets.items():
        for dx, dy in neighbor_offsets:
            nb = (kx + dx, ky + dy)
            if nb not in buckets:
                continue
            jdxs = buckets[nb]

            if dx == 0 and dy == 0:
                L = idxs
                m = len(L)
                for a in range(m):
                    i = L[a]
                    for b in range(a + 1, m):
                        j = L[b]
                        dxv = xs[i] - xs[j]
                        dyv = ys[i] - ys[j]
                        d2 = dxv * dxv + dyv * dyv
                        if d2 <= thr2:
                            mx = 0.5 * (xs[i] + xs[j])
                            my = 0.5 * (ys[i] + ys[j])
                            dist = float(np.sqrt(d2))
                            out.append((float(mx), float(my), dist, int(ids[i]), int(ids[j])))
            else:
                for i in idxs:
                    for j in jdxs:
                        dxv = xs[i] - xs[j]
                        dyv = ys[i] - ys[j]
                        d2 = dxv * dxv + dyv * dyv
                        if d2 <= thr2:
                            mx = 0.5 * (xs[i] + xs[j])
                            my = 0.5 * (ys[i] + ys[j])
                            dist = float(np.sqrt(d2))
                            out.append((float(mx), float(my), dist, int(ids[i]), int(ids[j])))

    return out

def extract_nearmiss_events(
    actor_df: pd.DataFrame,
    method: str,
    lead: int,
    rep: int,
    switch_pf: int,
    threshold_m: float,
    actor_type_prefix: str,
    keep_every_k_frames: int,
    max_pairs_per_frame: int,
) -> pd.DataFrame:
    df = actor_df.copy()
    if actor_type_prefix:
        df = df[df["actor_type"].str.lower().str.startswith(actor_type_prefix.lower())]

    df = df[df["payload_frame"] >= switch_pf]
    if df.empty:
        return pd.DataFrame()

    if keep_every_k_frames > 1:
        df = df[(df["payload_frame"] - switch_pf) % keep_every_k_frames == 0]
        if df.empty:
            return pd.DataFrame()

    events: List[Dict[str, object]] = []
    for pf, sub in df.groupby("payload_frame", sort=True):
        xs = sub["x"].to_numpy(dtype=np.float64, copy=False)
        ys = sub["y"].to_numpy(dtype=np.float64, copy=False)
        ids = sub["actor_id"].to_numpy(dtype=np.int64, copy=False)

        pairs = near_miss_pairs_midpoints(xs, ys, ids, threshold_m=threshold_m)
        if not pairs:
            continue

        # 1フレームの行数爆発を防ぐオプション（distが短い順に残す）
        if max_pairs_per_frame > 0 and len(pairs) > max_pairs_per_frame:
            pairs.sort(key=lambda t: t[2])  # dist
            pairs = pairs[:max_pairs_per_frame]

        for (mx, my, dist, ida, idb) in pairs:
            events.append({
                "method": method,
                "lead_sec": int(lead),
                "rep": int(rep),
                "payload_frame": int(pf),
                "x": float(mx),
                "y": float(my),
                "event_type": "near_miss",
                "source": "actor.csv",
                "min_dist_m": float(dist),
                "actor_id": int(ida),
                "other_id": int(idb),
                "intensity": np.nan,
                "is_accident": np.nan,
            })

    return pd.DataFrame(events)

# -------------------------------------------------------
# Collisions extraction + dedup symmetric (A,B) vs (B,A)
# -------------------------------------------------------
def canonical_collision_key(df: pd.DataFrame) -> pd.Series:
    a = df["actor_id"].astype("Int64")
    b = df["other_id"].astype("Int64")
    lo = a.where(a <= b, b)
    hi = b.where(a <= b, a)

    base = df["payload_frame"].astype("Int64").astype(str) + "|" + lo.astype("Int64").astype(str) + "|" + hi.astype("Int64").astype(str)
    if "intensity" not in df.columns:
        return base
    inten = pd.to_numeric(df["intensity"], errors="coerce").fillna(-1).round(3).astype(str)
    return base + "|" + inten

def extract_collision_events(
    coll_df: pd.DataFrame,
    method: str,
    lead: int,
    rep: int,
    switch_pf: int,
    only_accident: bool,
    min_intensity: Optional[float],
) -> pd.DataFrame:
    df = coll_df.copy()
    df = df[df["payload_frame"] >= switch_pf]
    if df.empty:
        return pd.DataFrame()

    if only_accident:
        if "is_accident" not in df.columns:
            raise SystemExit("--only-accident was set but is_accident not found in collisions.csv")
        df["is_accident"] = df["is_accident"].fillna(0).astype(int)
        df = df[df["is_accident"] == 1]
        if df.empty:
            return pd.DataFrame()

    if min_intensity is not None:
        if "intensity" not in df.columns:
            raise SystemExit("--min-intensity was set but intensity not found in collisions.csv")
        df = df[df["intensity"].fillna(-1) >= float(min_intensity)]
        if df.empty:
            return pd.DataFrame()

    df["__key"] = canonical_collision_key(df)
    df = df.drop_duplicates(subset=["__key"]).copy()

    out = pd.DataFrame({
        "method": method,
        "lead_sec": int(lead),
        "rep": int(rep),
        "payload_frame": df["payload_frame"].astype(int),
        "x": df["x"].astype(float),
        "y": df["y"].astype(float),
        "event_type": "collision",
        "source": "collisions.csv",
        "min_dist_m": np.nan,
        "actor_id": df["actor_id"].astype(int),
        "other_id": df["other_id"].astype(int),
        "intensity": df["intensity"].astype(float) if "intensity" in df.columns else np.nan,
        "is_accident": df["is_accident"].astype(int) if "is_accident" in df.columns else np.nan,
    })
    return out.reset_index(drop=True)

# -------------------------------------------------------
# Hotspot binning
# -------------------------------------------------------
def make_hotspot_bins(events: pd.DataFrame, group_cols: List[str], bin_size_m: float, topk: int) -> pd.DataFrame:
    if events.empty:
        cols = group_cols + ["bx","by","x_center","y_center","count"]
        return pd.DataFrame(columns=cols)

    bs = float(bin_size_m)
    ev = events.copy()
    ev["bx"] = np.floor(ev["x"].to_numpy(dtype=np.float64) / bs).astype(int)
    ev["by"] = np.floor(ev["y"].to_numpy(dtype=np.float64) / bs).astype(int)

    g = ev.groupby(group_cols + ["bx", "by"], as_index=False).size().rename(columns={"size": "count"})
    g["x_center"] = (g["bx"].astype(float) + 0.5) * bs
    g["y_center"] = (g["by"].astype(float) + 0.5) * bs

    g = g.sort_values(group_cols + ["count"], ascending=[True]*len(group_cols) + [False])
    if topk > 0:
        g["rank"] = g.groupby(group_cols).cumcount() + 1
        g = g[g["rank"] <= topk].drop(columns=["rank"])

    return g

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results_grid_accident")
    ap.add_argument("--outdir", type=str, default="", help="未指定なら新規生成")

    ap.add_argument("--methods", type=str, nargs="*", default=None)
    ap.add_argument("--lead", type=int, default=-1)
    ap.add_argument("--rep", type=int, default=-1)

    ap.add_argument("--base-payload-frame", type=int, default=25411)
    ap.add_argument("--dt", type=float, default=0.1)

    ap.add_argument("--threshold-m", type=float, default=5.0)
    ap.add_argument("--actor-type-prefix", type=str, default="vehicle")
    ap.add_argument("--keep-every-k-frames", type=int, default=1)

    ap.add_argument("--max-pairs-per-frame", type=int, default=-1,
                    help="near_missの1フレーム当たりの最大ペア数（distが短い順に残す）. -1=無制限")

    ap.add_argument("--include-nearmiss", action="store_true")
    ap.add_argument("--include-collisions", action="store_true")
    ap.add_argument("--only-accident", action="store_true")
    ap.add_argument("--min-intensity", type=float, default=None)

    ap.add_argument("--bin-size-m", type=float, default=5.0)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--per-run", action="store_true")
    args = ap.parse_args()

    if not args.include_nearmiss and not args.include_collisions:
        args.include_nearmiss = True
        args.include_collisions = True

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

    # New output dir
    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = root / f"out_hotspots_events_pairs_{ts}"
        outdir.mkdir(parents=True, exist_ok=True)

    if args.per_run:
        (outdir / "per_run").mkdir(exist_ok=True)

    all_events: List[pd.DataFrame] = []
    nm_all: List[pd.DataFrame] = []
    co_all: List[pd.DataFrame] = []

    for run in runs:
        switch_pf = int(args.base_payload_frame - run.lead * fps_i)

        actor_df = load_actor_csv(run.actor_csv)
        coll_df = load_collisions_csv(run.collisions_csv)

        nm = pd.DataFrame()
        co = pd.DataFrame()

        if args.include_nearmiss:
            nm = extract_nearmiss_events(
                actor_df=actor_df,
                method=run.method,
                lead=run.lead,
                rep=run.rep,
                switch_pf=switch_pf,
                threshold_m=args.threshold_m,
                actor_type_prefix=args.actor_type_prefix,
                keep_every_k_frames=max(1, int(args.keep_every_k_frames)),
                max_pairs_per_frame=int(args.max_pairs_per_frame),
            )
            if not nm.empty:
                nm_all.append(nm)
                all_events.append(nm)

        if args.include_collisions:
            co = extract_collision_events(
                coll_df=coll_df,
                method=run.method,
                lead=run.lead,
                rep=run.rep,
                switch_pf=switch_pf,
                only_accident=args.only_accident,
                min_intensity=args.min_intensity,
            )
            if not co.empty:
                co_all.append(co)
                all_events.append(co)

        if args.per_run:
            if args.include_nearmiss:
                nm.to_csv(outdir / "per_run" / f"events_nearmiss_{run.method}_lead{run.lead:02d}_rep{run.rep:02d}.csv", index=False)
            if args.include_collisions:
                co.to_csv(outdir / "per_run" / f"events_collision_{run.method}_lead{run.lead:02d}_rep{run.rep:02d}.csv", index=False)

        print(
            f"[OK] {run.method} lead={run.lead} rep={run.rep} "
            f"near_miss_pairs={len(nm) if not nm.empty else 0} "
            f"collision={len(co) if not co.empty else 0}"
        )

    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    nm_df = pd.concat(nm_all, ignore_index=True) if nm_all else pd.DataFrame()
    co_df = pd.concat(co_all, ignore_index=True) if co_all else pd.DataFrame()

    events.to_csv(outdir / "events_all.csv", index=False)
    nm_df.to_csv(outdir / "events_nearmiss.csv", index=False)
    co_df.to_csv(outdir / "events_collision.csv", index=False)

    # Hotspots (union)
    make_hotspot_bins(events, ["method"], args.bin_size_m, args.topk).to_csv(outdir / "hotspots_by_method.csv", index=False)
    make_hotspot_bins(events, ["method", "lead_sec"], args.bin_size_m, args.topk).to_csv(outdir / "hotspots_by_method_lead.csv", index=False)
    make_hotspot_bins(events, ["method", "lead_sec", "rep"], args.bin_size_m, args.topk).to_csv(outdir / "hotspots_by_method_lead_rep.csv", index=False)

    # Split views: by event_type / by source
    make_hotspot_bins(events, ["method", "event_type"], args.bin_size_m, args.topk).to_csv(outdir / "hotspots_by_method_event.csv", index=False)
    make_hotspot_bins(events, ["method", "source"], args.bin_size_m, args.topk).to_csv(outdir / "hotspots_by_method_source.csv", index=False)

    print(f"[OK] 出力先: {outdir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
