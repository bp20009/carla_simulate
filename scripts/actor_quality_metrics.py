#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import carla


@dataclass(frozen=True)
class RunKey:
    method: str
    lead_sec: int
    rep: int


def load_json(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def safe_int(v: object) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(float(str(v).strip()))
    except Exception:
        return None


def safe_float(v: object) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(str(v).strip())
    except Exception:
        return None


def parse_run_key_from_path(p: Path) -> Optional[RunKey]:
    # .../<method>/lead_<N>/rep_<N>/logs/actor.csv
    parts = p.parts
    method = None
    lead = None
    rep = None
    for part in parts:
        if part in ("autopilot", "lstm", "none"):
            method = part
        m = re.fullmatch(r"lead_(\d+)", part)
        if m:
            lead = int(m.group(1))
        m = re.fullmatch(r"rep_(\d+)", part)
        if m:
            rep = int(m.group(1))
    if method is None or lead is None or rep is None:
        return None
    return RunKey(method=method, lead_sec=lead, rep=rep)


def iter_runs(root: Path) -> Iterator[Tuple[RunKey, Path, Path]]:
    # yield (key, actor_csv, meta_json)
    for apath in root.glob("**/logs/actor.csv"):
        key = parse_run_key_from_path(apath)
        if key is None:
            continue
        mpath = apath.parent / "meta.json"
        if not mpath.exists():
            continue
        yield key, apath, mpath


def unwrap_deg(prev: float, cur: float) -> float:
    # keep continuity in degrees
    d = cur - prev
    while d > 180.0:
        cur -= 360.0
        d = cur - prev
    while d < -180.0:
        cur += 360.0
        d = cur - prev
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="e.g., results_grid_accident")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--dt", type=float, default=0.1, help="fixed delta seconds (default 0.1)")
    ap.add_argument("--vehicle-only", action="store_true", help="only evaluate vehicle.* actors")
    ap.add_argument("--lane-dist-th", type=float, default=2.0, help="lane-center distance threshold meters")
    ap.add_argument("--accel-th", type=float, default=3.0, help="|accel| threshold m/s^2")
    ap.add_argument("--yawrate-th-deg", type=float, default=30.0, help="|yaw_rate| threshold deg/s")
    ap.add_argument("--out-run", type=str, default="actor_metrics_per_run.csv")
    ap.add_argument("--out-agg", type=str, default="actor_metrics_by_method_lead.csv")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()

    # Connect to CARLA (for map waypoint queries)
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    cmap = world.get_map()

    out_run = Path(args.out_run).expanduser().resolve()
    out_run.parent.mkdir(parents=True, exist_ok=True)
    out_agg = Path(args.out_agg).expanduser().resolve()
    out_agg.parent.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, object]] = []

    for key, actor_csv, meta_json in iter_runs(root):
        meta = load_json(meta_json)
        switch_pf = safe_int(meta.get("switch_payload_frame_observed"))
        if switch_pf is None:
            # 厳密に「予測開始後」を切れないのでスキップ
            continue

        # 予測開始後の時系列を actor_id ごとに蓄える
        # {actor_id: [(frame, x, y, z, yaw_deg)]}
        tracks: Dict[int, List[Tuple[int, float, float, float, float]]] = {}

        # 走行逸脱（point-wise）
        n_points = 0
        n_offroad = 0
        lane_dists: List[float] = []
        n_lane_over = 0

        with actor_csv.open(encoding="utf-8", errors="ignore", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                frame = safe_int(row.get("frame"))
                if frame is None or frame < switch_pf:
                    continue

                actor_type = (row.get("type") or "").strip()
                if args.vehicle_only and not actor_type.startswith("vehicle."):
                    continue

                actor_id = safe_int(row.get("id"))
                x = safe_float(row.get("location_x"))
                y = safe_float(row.get("location_y"))
                z = safe_float(row.get("location_z"))
                yaw = safe_float(row.get("rotation_yaw"))
                if actor_id is None or x is None or y is None or z is None or yaw is None:
                    continue

                # point-wise lane/offroad check
                loc = carla.Location(x=float(x), y=float(y), z=float(z))
                wp_no = cmap.get_waypoint(loc, project_to_road=False)
                if wp_no is None:
                    n_offroad += 1

                wp = cmap.get_waypoint(loc, project_to_road=True)
                if wp is not None:
                    dx = float(loc.x - wp.transform.location.x)
                    dy = float(loc.y - wp.transform.location.y)
                    dz = float(loc.z - wp.transform.location.z)
                    d = math.sqrt(dx * dx + dy * dy + dz * dz)
                    lane_dists.append(d)
                    if d >= float(args.lane_dist_th):
                        n_lane_over += 1

                n_points += 1

                tracks.setdefault(actor_id, []).append((frame, float(x), float(y), float(z), float(yaw)))

        # speed/accel/yawrate anomaly (per-actor consecutive frames)
        # reconstruct v from position diff
        accel_hits = 0
        accel_total = 0
        yaw_hits = 0
        yaw_total = 0

        for aid, seq in tracks.items():
            seq.sort(key=lambda t: t[0])

            # unwrap yaw in-place
            unwrapped: List[Tuple[int, float, float, float, float]] = []
            prev_yaw = None
            cur_yaw = None
            for fr, x, y, z, yaw in seq:
                if prev_yaw is None:
                    cur_yaw = yaw
                else:
                    cur_yaw = unwrap_deg(prev_yaw, yaw)
                unwrapped.append((fr, x, y, z, float(cur_yaw)))
                prev_yaw = float(cur_yaw)

            # speed list
            speeds: List[Tuple[int, float]] = []
            for i in range(1, len(unwrapped)):
                f0, x0, y0, z0, _ = unwrapped[i - 1]
                f1, x1, y1, z1, _ = unwrapped[i]
                df = f1 - f0
                if df <= 0:
                    continue
                dt = df * float(args.dt)
                dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
                v = dist / dt
                speeds.append((f1, v))

            # accel
            for i in range(1, len(speeds)):
                _, v0 = speeds[i - 1]
                _, v1 = speeds[i]
                a = (v1 - v0) / float(args.dt)
                accel_total += 1
                if abs(a) >= float(args.accel_th):
                    accel_hits += 1

            # yaw rate
            for i in range(1, len(unwrapped)):
                f0, _, _, _, yaw0 = unwrapped[i - 1]
                f1, _, _, _, yaw1 = unwrapped[i]
                df = f1 - f0
                if df <= 0:
                    continue
                dt = df * float(args.dt)
                yr = (yaw1 - yaw0) / dt  # deg/s
                yaw_total += 1
                if abs(yr) >= float(args.yawrate_th_deg):
                    yaw_hits += 1

        lane_mean = (sum(lane_dists) / len(lane_dists)) if lane_dists else None
        lane_p95 = None
        if lane_dists:
            s = sorted(lane_dists)
            idx = int(math.floor(0.95 * (len(s) - 1)))
            lane_p95 = s[idx]

        run_rows.append(
            {
                "method": key.method,
                "lead_sec": key.lead_sec,
                "rep": key.rep,
                "switch_pf": switch_pf,
                "points": n_points,
                "offroad_points": n_offroad,
                "offroad_rate": (n_offroad / n_points) if n_points > 0 else "",
                "lane_samples": len(lane_dists),
                "lane_dist_mean_m": lane_mean if lane_mean is not None else "",
                "lane_dist_p95_m": lane_p95 if lane_p95 is not None else "",
                "lane_over_th_rate": (n_lane_over / len(lane_dists)) if lane_dists else "",
                "accel_samples": accel_total,
                "accel_abnormal_rate": (accel_hits / accel_total) if accel_total > 0 else "",
                "yawrate_samples": yaw_total,
                "yawrate_abnormal_rate": (yaw_hits / yaw_total) if yaw_total > 0 else "",
                "actor_csv": str(actor_csv),
            }
        )

    # write per-run
    run_fields = [
        "method",
        "lead_sec",
        "rep",
        "switch_pf",
        "points",
        "offroad_points",
        "offroad_rate",
        "lane_samples",
        "lane_dist_mean_m",
        "lane_dist_p95_m",
        "lane_over_th_rate",
        "accel_samples",
        "accel_abnormal_rate",
        "yawrate_samples",
        "yawrate_abnormal_rate",
        "actor_csv",
    ]
    with out_run.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=run_fields)
        w.writeheader()
        for row in run_rows:
            w.writerow(row)

    # aggregate by method x lead
    def agg_key(r: Dict[str, object]) -> Tuple[str, int]:
        return (str(r["method"]), int(r["lead_sec"]))

    groups: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for r in run_rows:
        groups.setdefault(agg_key(r), []).append(r)

    agg_fields = [
        "method",
        "lead_sec",
        "runs",
        "offroad_rate_mean",
        "lane_over_th_rate_mean",
        "lane_dist_mean_m_mean",
        "accel_abnormal_rate_mean",
        "yawrate_abnormal_rate_mean",
    ]
    with out_agg.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        for (method, lead), rows in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
            def mean_of(field: str) -> Optional[float]:
                vals = []
                for rr in rows:
                    v = rr.get(field)
                    if v == "" or v is None:
                        continue
                    try:
                        vals.append(float(v))
                    except Exception:
                        pass
                if not vals:
                    return None
                return sum(vals) / len(vals)

            w.writerow(
                {
                    "method": method,
                    "lead_sec": lead,
                    "runs": len(rows),
                    "offroad_rate_mean": mean_of("offroad_rate") or "",
                    "lane_over_th_rate_mean": mean_of("lane_over_th_rate") or "",
                    "lane_dist_mean_m_mean": mean_of("lane_dist_mean_m") or "",
                    "accel_abnormal_rate_mean": mean_of("accel_abnormal_rate") or "",
                    "yawrate_abnormal_rate_mean": mean_of("yawrate_abnormal_rate") or "",
                }
            )

    print(f"Wrote per-run: {out_run}")
    print(f"Wrote aggregate: {out_agg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
