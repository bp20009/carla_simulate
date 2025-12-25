#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple


@dataclass
class RunKey:
    method: str
    lead_sec: int
    rep: int


def load_json(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
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


def parse_run_key_from_collisions_path(collisions_path: Path) -> Optional[RunKey]:
    """
    Expected:
      .../<method>/lead_<N>/rep_<N>/logs/collisions.csv
    """
    parts = collisions_path.parts
    method = None
    lead = None
    rep = None

    for p in parts:
        if p in ("autopilot", "lstm", "none"):
            method = p
        m = re.fullmatch(r"lead_(\d+)", p)
        if m:
            lead = int(m.group(1))
        m = re.fullmatch(r"rep_(\d+)", p)
        if m:
            rep = int(m.group(1))

    if method is None or lead is None or rep is None:
        return None
    return RunKey(method=method, lead_sec=lead, rep=rep)


def iter_runs(root: Path) -> Iterator[Tuple[RunKey, Path, Path]]:
    """
    Yield (run_key, collisions_csv_path, meta_json_path)
    """
    for cpath in root.glob("**/logs/collisions.csv"):
        key = parse_run_key_from_collisions_path(cpath)
        if key is None:
            continue
        mpath = cpath.parent / "meta.json"
        if not mpath.exists():
            continue
        yield key, cpath, mpath


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="e.g., results_grid_accident")
    ap.add_argument("--threshold", type=float, default=1000.0, help="intensity threshold (default 1000)")
    ap.add_argument("--out", type=str, default="future_accidents_ge1000.csv")
    ap.add_argument("--require-is-accident", action="store_true", help="require is_accident==1")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "lead_sec",
        "rep",
        "switch_payload_frame_observed",
        "payload_frame",
        "carla_frame",
        "actor_id",
        "actor_type",
        "other_id",
        "other_type",
        "x",
        "y",
        "z",
        "intensity",
        "collisions_csv",
    ]

    total_runs = 0
    hit_runs = 0
    total_hits = 0

    with out_path.open("w", encoding="utf-8", newline="") as out_f:
        w = csv.DictWriter(out_f, fieldnames=fieldnames)
        w.writeheader()

        for key, collisions_csv, meta_json in iter_runs(root):
            total_runs += 1
            meta = load_json(meta_json)
            switch_pf = safe_int(meta.get("switch_payload_frame_observed"))

            # switch_pf が無い場合，「予測時」の切り分けができないのでスキップ（厳密）
            if switch_pf is None:
                continue

            wrote_any = False
            with collisions_csv.open(encoding="utf-8", errors="ignore", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    intensity = safe_float(row.get("intensity"))
                    if intensity is None or intensity < float(args.threshold):
                        continue

                    if args.require_is_accident and "is_accident" in row:
                        if str(row.get("is_accident", "")).strip() != "1":
                            continue

                    payload_frame = safe_int(row.get("payload_frame"))
                    if payload_frame is None:
                        continue

                    # 予測時（future開始後）に限定
                    if payload_frame < switch_pf:
                        continue

                    wrote_any = True
                    total_hits += 1
                    w.writerow(
                        {
                            "method": key.method,
                            "lead_sec": key.lead_sec,
                            "rep": key.rep,
                            "switch_payload_frame_observed": switch_pf,
                            "payload_frame": payload_frame,
                            "carla_frame": safe_int(row.get("carla_frame")) or "",
                            "actor_id": safe_int(row.get("actor_id")) or "",
                            "actor_type": row.get("actor_type", "") or "",
                            "other_id": safe_int(row.get("other_id")) or "",
                            "other_type": row.get("other_type", "") or "",
                            "x": safe_float(row.get("x")) or "",
                            "y": safe_float(row.get("y")) or "",
                            "z": safe_float(row.get("z")) or "",
                            "intensity": intensity,
                            "collisions_csv": str(collisions_csv),
                        }
                    )

            if wrote_any:
                hit_runs += 1

    print(f"Scanned runs (with meta+collisions): {total_runs}")
    print(f"Runs with extracted hits: {hit_runs}")
    print(f"Total extracted events: {total_hits}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
